//! Concurrency stress tests for `--features parallel`.
//!
//! Why this file exists
//! --------------------
//! The `parallel` feature turns `ExprPool`'s deduplication index into a sharded
//! `DashMap` and lets Rayon workers — and, through PyO3's `allow_threads`,
//! arbitrary Python threads — intern into one pool concurrently.  Everything
//! that made that sound was argued in comments and never executed:
//!
//!   * the only pool-concurrency coverage in the crate was *indirect*, through
//!     `simplify_par` / `simplify_redex` unit tests that happen to run on Rayon;
//!   * `alkahest-core/examples/pool_{read,build}_scaling.rs` do hammer a shared
//!     pool from many threads, but they are examples, so no CI job runs them;
//!   * the nightly ThreadSanitizer shard built the workspace with *default*
//!     features, i.e. with `parallel` off, so it never compiled — let alone
//!     executed — a single line of the code above.
//!
//! These tests are written to be run twice: once normally (`cargo test
//! --features parallel`, where they check invariants) and once under
//! ThreadSanitizer (the nightly `tsan` shard, where the *access pattern* is the
//! product and the assertions are secondary).  They deliberately use raw
//! `std::thread` rather than Rayon so the threads are genuine OS threads that
//! do not cooperate through a work-stealing scheduler — that is what a Python
//! caller holding one `ExprPool` across several `threading.Thread`s looks like.
//!
//! Keep them fast: TSan is ~10× slower and this file runs inside a 6 h shard
//! shared with ASan, LSan and Valgrind.

#![cfg(feature = "parallel")]

use alkahest_cas::{
    compile, simplify, simplify_par, simplify_redex, CompiledFn, Domain, ExprId, ExprPool,
};
use std::collections::HashSet;
use std::sync::{Arc, Barrier};
use std::thread;

/// Threads per test.  Enough to have several DashMap shards contended at once
/// without oversubscribing a 2-core GitHub runner into uselessness.
const THREADS: usize = 8;

/// Run `body` on `THREADS` threads that all start at the same instant.
///
/// The barrier matters: without it the first thread usually finishes its whole
/// workload before the last one is scheduled, every intern hits the read-only
/// fast path, and the test passes without ever exercising a contended shard —
/// a green tick for a code path that never ran.
fn in_lockstep<T, F>(body: F) -> Vec<T>
where
    T: Send + 'static,
    F: Fn(usize) -> T + Send + Sync + 'static,
{
    let barrier = Arc::new(Barrier::new(THREADS));
    let body = Arc::new(body);
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let barrier = Arc::clone(&barrier);
            let body = Arc::clone(&body);
            thread::spawn(move || {
                barrier.wait();
                body(t)
            })
        })
        .collect();
    handles
        .into_iter()
        .map(|h| h.join().expect("worker thread panicked"))
        .collect()
}

/// A deterministic, moderately deep expression: `((x + k)*(x + k + 1))^2 + k`.
fn build_shape(pool: &ExprPool, x: ExprId, k: i64) -> ExprId {
    let a = pool.add(vec![x, pool.integer(k)]);
    let b = pool.add(vec![x, pool.integer(k + 1)]);
    let prod = pool.mul(vec![a, b]);
    let sq = pool.pow(prod, pool.integer(2));
    pool.add(vec![sq, pool.integer(k)])
}

// ---------------------------------------------------------------------------
// Interning
// ---------------------------------------------------------------------------

/// Hash-consing must survive contention: identical structure built on N threads
/// at once must yield one node, not N.
///
/// This is the sharp test of `DashMap::entry(..).or_insert_with(..)` wrapping
/// the `boxcar::push`.  If that were a check-then-act (`get`, then `insert`)
/// two threads could race to push the same `ExprData` twice; both would get a
/// *valid* id, nothing would panic, and structural equality would silently stop
/// implying id equality — the invariant every downstream `==` in this crate
/// relies on.  `pool.len()` is the detector: it counts pushes, so a duplicate
/// push shows up as a node count above the single-threaded baseline.
#[test]
fn concurrent_interning_preserves_hash_consing() {
    const SHAPES: i64 = 120;

    // Single-threaded baseline for the exact same work.
    let baseline_pool = ExprPool::new();
    let bx = baseline_pool.symbol("x", Domain::Real);
    let baseline_ids: Vec<ExprId> = (0..SHAPES)
        .map(|k| build_shape(&baseline_pool, bx, k))
        .collect();
    let baseline_len = baseline_pool.len();

    let pool = Arc::new(ExprPool::new());
    let x = pool.symbol("x", Domain::Real);
    let shared = Arc::clone(&pool);

    let per_thread = in_lockstep(move |_t| {
        (0..SHAPES)
            .map(|k| build_shape(&shared, x, k))
            .collect::<Vec<_>>()
    });

    // Every thread must have observed the same ids as every other thread...
    for ids in &per_thread {
        assert_eq!(
            ids, &per_thread[0],
            "identical structures interned to different ids on different threads: \
             hash-consing is not atomic under contention"
        );
    }
    // ...and no duplicate node may have been pushed.
    assert_eq!(
        pool.len(),
        baseline_len,
        "concurrent interning created {} extra node(s) versus the single-threaded \
         baseline — a duplicate push means two threads both ran the or_insert_with \
         closure for one key",
        pool.len().saturating_sub(baseline_len)
    );
    // And the shared pool agrees structurally with the baseline pool.
    for (i, id) in per_thread[0].iter().enumerate() {
        assert_eq!(
            pool.get(*id),
            baseline_pool.get(baseline_ids[i]),
            "node {i} differs between the concurrent and sequential pools"
        );
    }
}

/// Distinct structures interned concurrently must all be present, distinct, and
/// readable — the `boxcar::Vec` reference-stability claim, exercised while the
/// array is actually growing under other threads' pushes.
#[test]
fn concurrent_interning_of_distinct_structures_keeps_every_node_readable() {
    const PER_THREAD: i64 = 80;

    let pool = Arc::new(ExprPool::new());
    let shared = Arc::clone(&pool);

    let per_thread = in_lockstep(move |t| {
        let x = shared.symbol(format!("x{t}"), Domain::Real);
        (0..PER_THREAD)
            .map(|k| {
                let id = build_shape(&shared, x, k);
                // Read it straight back on this thread while everyone else is
                // still pushing: `depth` and `is_mult_commutative` are cached
                // fields written inside the shard lock and read lock-free.
                assert_eq!(
                    shared.depth(id),
                    5,
                    "cached depth wrong for k={k} on thread {t}"
                );
                assert!(shared.is_mult_commutative(id));
                id
            })
            .collect::<Vec<_>>()
    });

    let all: HashSet<ExprId> = per_thread.iter().flatten().copied().collect();
    assert_eq!(
        all.len(),
        THREADS * PER_THREAD as usize,
        "distinct structures collided on an id"
    );
    // Every id, from every thread, must still resolve after the joins.
    for id in &all {
        assert_eq!(pool.depth(*id), 5);
    }
}

/// Readers must never observe a half-published node.
///
/// `ExprPool::with` hands out a `&ExprData` borrowed straight out of the
/// `boxcar::Vec` with no lock at all, while writers are pushing.  If the node
/// array published a slot before its `ExprData` was fully written, this is
/// where a torn read would surface — as a wrong value here, or as a TSan report
/// on the nightly shard.
#[test]
fn lock_free_reads_are_consistent_while_writers_intern() {
    const ROUNDS: i64 = 200;

    let pool = Arc::new(ExprPool::new());
    let x = pool.symbol("x", Domain::Real);
    // A set of ids published before any thread starts, so readers always have
    // something legitimate to look at.
    let seeded: Vec<ExprId> = (0..32).map(|k| build_shape(&pool, x, -k - 1)).collect();
    let seeded_data: Vec<_> = seeded.iter().map(|id| pool.get(*id)).collect();

    let shared = Arc::clone(&pool);
    let seeded = Arc::new(seeded);
    let seeded_data = Arc::new(seeded_data);

    in_lockstep(move |t| {
        if t % 2 == 0 {
            // Writer.
            for k in 0..ROUNDS {
                build_shape(&shared, x, k + t as i64 * ROUNDS);
            }
        } else {
            // Reader: lock-free `with` / `len` against a growing array.
            for _ in 0..ROUNDS {
                for (i, id) in seeded.iter().enumerate() {
                    shared.with(*id, |d| {
                        assert_eq!(*d, seeded_data[i], "torn read of a published node");
                    });
                }
                // `len()` must be monotone and never index past the array.
                let n = shared.len();
                assert!(n >= seeded.len());
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Parallel simplification entry points
// ---------------------------------------------------------------------------

/// `simplify_par` called from many OS threads against **one shared pool**.
///
/// This is the shape the PyO3 binding creates: `py_simplify_par` takes a
/// `PyRef<PyExprPool>`, strips it to `&ExprPool`, and calls this under
/// `allow_threads` — so N Python threads holding the same pool object put N
/// concurrent `simplify_par` calls on it, each of which itself forks onto
/// Rayon and interns its intermediate results into that same pool.
#[test]
fn simplify_par_from_many_threads_on_one_pool_matches_sequential() {
    let pool = Arc::new(ExprPool::new());
    let x = pool.symbol("x", Domain::Real);

    // Sequential answers first, computed alone, as the oracle.
    let inputs: Vec<ExprId> = (0..THREADS as i64)
        .map(|k| build_shape(&pool, x, k))
        .collect();
    let expected: Vec<ExprId> = inputs.iter().map(|e| simplify(*e, &pool).value).collect();

    let shared = Arc::clone(&pool);
    let inputs = Arc::new(inputs);
    let expected = Arc::new(expected);

    in_lockstep(move |t| {
        for round in 0..4usize {
            // Rotate so threads collide on the same expression in some rounds
            // and diverge in others.
            let i = (t + round) % inputs.len();
            let got = simplify_par(inputs[i], &shared).value;
            assert_eq!(
                got, expected[i],
                "simplify_par disagreed with simplify on thread {t}, round {round}"
            );
        }
    });
}

/// Same, for the level-scheduled scheduler.  It keeps its memo in a
/// `Vec<AtomicU32>` sized to `pool.len()` at pass start, while other threads are
/// concurrently interning ids *beyond* that range — so its bound is only safe
/// because those ids are stored as values and never used as keys.  Running two
/// of these against one growing pool is the test of that argument.
#[test]
fn simplify_redex_from_many_threads_on_one_pool_matches_sequential() {
    let pool = Arc::new(ExprPool::new());
    let x = pool.symbol("x", Domain::Real);

    let inputs: Vec<ExprId> = (0..THREADS as i64)
        .map(|k| build_shape(&pool, x, k * 7))
        .collect();
    let expected: Vec<ExprId> = inputs.iter().map(|e| simplify(*e, &pool).value).collect();

    let shared = Arc::clone(&pool);
    let inputs = Arc::new(inputs);
    let expected = Arc::new(expected);

    in_lockstep(move |t| {
        for round in 0..4usize {
            let i = (t + round) % inputs.len();
            let got = simplify_redex(inputs[i], &shared).value;
            assert_eq!(
                got, expected[i],
                "simplify_redex disagreed with simplify on thread {t}, round {round}"
            );
        }
    });
}

/// Simplifying on some threads while others intern into the same pool.
///
/// Nothing stops a Python program doing this — `pool.symbol(...)` on one thread
/// while `simplify_par(expr)` runs GIL-free on another — and it is the one
/// combination neither the sequential tests nor the Rayon-internal ones cover.
#[test]
fn interning_concurrently_with_simplify_par_is_safe() {
    let pool = Arc::new(ExprPool::new());
    let x = pool.symbol("x", Domain::Real);
    let target = build_shape(&pool, x, 3);
    let expected = simplify(target, &pool).value;

    let shared = Arc::clone(&pool);
    in_lockstep(move |t| {
        if t % 2 == 0 {
            for _ in 0..4 {
                assert_eq!(simplify_par(target, &shared).value, expected);
            }
        } else {
            let y = shared.symbol(format!("y{t}"), Domain::Real);
            for k in 0..150 {
                let id = build_shape(&shared, y, k);
                assert_eq!(shared.depth(id), 5);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// numpy_eval_par's Rust core
// ---------------------------------------------------------------------------

/// `CompiledFn::call_batch_par` is what `numpy_eval_par` runs inside
/// `allow_threads`.  `CompiledFn` carries its own `unsafe impl Send + Sync`
/// (justified by the code pages being read-only after finalize), so sharing one
/// across threads while each calls `call_batch_par` — itself a Rayon fan-out —
/// is the nested case that assertion has to hold up under.
#[test]
fn call_batch_par_is_reentrant_across_threads() {
    const POINTS: usize = 512;

    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let expr = build_shape(&pool, x, 1);
    let f: Arc<CompiledFn> = Arc::new(compile(expr, &[x], &pool).expect("compile failed"));

    let xs: Arc<Vec<f64>> = Arc::new((0..POINTS).map(|i| i as f64 / POINTS as f64).collect());

    // Sequential oracle.
    let mut want = vec![0.0f64; POINTS];
    f.call_batch(&[&xs[..]], &mut want);
    let want = Arc::new(want);

    in_lockstep(move |t| {
        let mut got = vec![f64::NAN; POINTS];
        f.call_batch_par(&[&xs[..]], &mut got);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 1e-9 * w.abs().max(1.0),
                "call_batch_par disagreed with call_batch at point {i} on thread {t}: \
                 {g} vs {w}"
            );
        }
    });
}
