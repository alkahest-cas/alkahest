//! A thread that used FLINT must not leave its caches behind when it exits.
//!
//! FLINT's `fmpz` cache is per thread: the first big integer a thread makes
//! allocates a 16-page block of `mpz_t` structs, every one initialised with
//! its own GMP allocation, and a cleared big integer returns to that cache
//! rather than to the allocator. Only `flint_cleanup` releases it, and nothing
//! called that, so every thread that ever made a big integer leaked ~228 kB
//! when it exited — 5000 short-lived threads grew the process by 1.14 GB.
//!
//! This is its own test binary, with a single test, so no concurrently running
//! test can move the resident-set size it measures.

#[cfg(target_os = "linux")]
fn status_kb(field: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    status
        .lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap()
}

/// `true` under ASan/TSan/MSan, which reserve terabytes of shadow address
/// space. ASan also quarantines freed memory instead of reusing it, so a
/// resident-set bound means nothing there; the fix is still exercised by the
/// unit tests that observe the release path directly.
#[cfg(target_os = "linux")]
fn under_sanitizer() -> bool {
    status_kb("VmSize:") > 1 << 30
}

#[cfg(target_os = "linux")]
#[test]
fn short_lived_threads_do_not_leak_flint_caches() {
    use alkahest_cas::flint::FlintInteger;

    fn work() {
        let big = FlintInteger::from_i64(3).pow(200);
        assert_eq!(big.to_string().len(), 96);
    }

    if under_sanitizer() {
        eprintln!("skipped: sanitizer build (freed memory is quarantined)");
        return;
    }
    const THREADS: u64 = 400;
    // Warm up the allocator arenas and this thread's own cache first, so the
    // measurement sees only what each exiting thread leaves behind.
    work();
    for _ in 0..32 {
        std::thread::spawn(work).join().unwrap();
    }
    let before = status_kb("VmRSS:");
    for _ in 0..THREADS {
        std::thread::spawn(work).join().unwrap();
    }
    let grown = status_kb("VmRSS:").saturating_sub(before);
    let per_thread = grown / THREADS;
    eprintln!("{THREADS} threads: RSS +{grown} kB ({per_thread} kB/thread)");
    // Unfixed, each thread leaves ~200 kB behind (+80 MB here). Fixed, the
    // growth is allocator noise. The bound sits far from both.
    assert!(
        per_thread < 50,
        "{THREADS} short-lived threads that each made one big integer grew RSS by \
         {grown} kB ({per_thread} kB per thread): FLINT's per-thread cache is not \
         being released at thread exit"
    );
}
