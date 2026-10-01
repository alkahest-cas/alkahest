//! Release FLINT's per-thread caches when a thread that used FLINT exits.
//!
//! # The leak
//!
//! FLINT keeps several caches in thread-local storage and frees them only in
//! `flint_cleanup`, which nothing in this crate called. The large one is the
//! `fmpz` cache (`fmpz_single.c`): the first time a thread promotes an `fmpz`
//! to a GMP integer, FLINT allocates a 16-page block of `mpz_t` structs,
//! initialises every one of them (a GMP allocation each) and threads them onto
//! a thread-local free list. A cleared big integer goes back on that list
//! rather than to the allocator. When the thread exits the list is simply
//! forgotten, and with it the block and every limb allocation behind it —
//! about 228 kB per thread. 5000 short-lived threads that each made one
//! `3^200` grew the process by 1.14 GB. Thread-per-request servers and any
//! pool that retires its workers pay that on every thread. `mpfr_free_cache`
//! (MPFR's constant caches) and Arb's per-thread constant caches are released
//! by the same call.
//!
//! # The fix
//!
//! [`note_thread_uses_flint`] touches a thread-local guard whose `Drop` calls
//! `flint_cleanup`. The FLINT wrappers call it from their constructors, so the
//! guard is registered on the first FLINT use of each thread and runs when the
//! thread exits. A thread that never touches FLINT registers nothing, and a
//! long-lived thread (Rayon's pool workers, the Python main thread) pays one
//! thread-local access per constructor and the cleanup once, at exit.
//!
//! # Why it is sound
//!
//! `flint_cleanup` (FLINT 3.3–3.5; a no-op in a reentrant build without TLS)
//! touches only the *calling* thread's caches:
//!
//! * The `mpz_t` structs on this thread's free list are cleared and counted
//!   against their block; a block is freed only once *every* struct in it has
//!   been counted. An integer that is still alive — held by another thread,
//!   or by a value this thread has not dropped yet — keeps its block alive.
//!   When it is eventually cleared, FLINT sees the block's non-zero count (or
//!   a foreign owning thread) and takes its cross-thread path: `mpz_clear`
//!   plus an atomic increment, freeing the block on the last one. That path
//!   is FLINT's own, used today for any integer dropped on a thread other
//!   than the one that made it.
//! * The thread-local pointers are reset to empty afterwards, so FLINT use
//!   *after* the cleanup (from a later thread-local destructor) starts a
//!   fresh cache instead of touching freed memory. That cache is then not
//!   released, which is the old behaviour, not a new hazard.
//! * GMP memory is returned through GMP's free function, which is the
//!   accounting wrapper of [`crate::budget::install_memory_accounting`] when
//!   installed. The wrapper only updates a process-wide atomic, so it is safe
//!   to run from a thread-local destructor.
//!
//! The guard is reached through `try_with`, so FLINT use from another
//! thread-local destructor that runs after this one is not a panic.

use super::ffi;

struct ThreadCleanup;

impl Drop for ThreadCleanup {
    fn drop(&mut self) {
        // SAFETY: see the module documentation — `flint_cleanup` releases
        // only the calling thread's caches and leaves them reusable.
        unsafe { ffi::flint_cleanup() };
        #[cfg(test)]
        {
            let tag = TAG.with(|t| t.get());
            if tag != 0 {
                CLEANED_TAGS.lock().unwrap().push(tag);
            }
        }
    }
}

thread_local! {
    static GUARD: ThreadCleanup = const { ThreadCleanup };
}

#[cfg(test)]
thread_local! {
    /// A test's label for this thread. A `Cell` of plain data has no
    /// destructor, so it stays readable while other thread-locals are being
    /// destroyed, in whatever order a platform runs them.
    static TAG: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Tags of the threads whose guard ran `flint_cleanup`.
#[cfg(test)]
static CLEANED_TAGS: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());

/// Register the calling thread for `flint_cleanup` when it exits.
///
/// Cheap and idempotent; call it before a thread's first FLINT allocation.
#[inline]
pub(crate) fn note_thread_uses_flint() {
    // Kani harnesses stub FLINT by its contract and model no threads.
    #[cfg(not(kani))]
    let _ = GUARD.try_with(|_| {});
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flint::FlintInteger;
    use rug::ops::Pow;

    /// A thread that made a big integer runs `flint_cleanup` on exit, and a
    /// thread that never touched FLINT does not.
    #[test]
    fn a_thread_that_used_flint_cleans_up_when_it_exits() {
        const USED: u64 = 0x5eed_0001;
        const IDLE: u64 = 0x5eed_0002;
        std::thread::spawn(|| {
            TAG.with(|t| t.set(USED));
            let big = FlintInteger::from_i64(3).pow(200);
            assert!(big.to_string().len() > 90);
        })
        .join()
        .unwrap();
        std::thread::spawn(|| TAG.with(|t| t.set(IDLE)))
            .join()
            .unwrap();
        let tags = CLEANED_TAGS.lock().unwrap();
        assert_eq!(tags.iter().filter(|&&t| t == USED).count(), 1);
        assert!(!tags.contains(&IDLE));
    }

    /// The cleanup must not free a block that still backs a live integer: a
    /// big integer made on a thread that has since exited (and cleaned up)
    /// stays valid, and is released through FLINT's cross-thread path.
    #[test]
    fn a_big_integer_outlives_the_thread_that_made_it() {
        let made: Vec<FlintInteger> = std::thread::spawn(|| {
            (0..100u64)
                .map(|i| FlintInteger::from_i64(3).pow(200 + i))
                .collect()
        })
        .join()
        .unwrap();
        // The maker has exited and run `flint_cleanup`. Reuse the freed
        // memory from fresh threads, then read and drop the survivors here.
        for _ in 0..8 {
            std::thread::spawn(|| {
                let v: Vec<FlintInteger> = (0..200)
                    .map(|_| FlintInteger::from_i64(7).pow(150))
                    .collect();
                assert_eq!(v.len(), 200);
            })
            .join()
            .unwrap();
        }
        for (i, n) in made.iter().enumerate() {
            let want = rug::Integer::from(3).pow(200 + i as u32);
            assert_eq!(n.to_rug(), want);
        }
        drop(made);
    }
}
