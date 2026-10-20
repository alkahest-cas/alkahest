//! Wall-clock allowances for unit tests (test builds only).
//!
//! A test that asserts "this finishes within N seconds" is checking an
//! *order of magnitude* (a linear walk of a shared DAG versus a walk of its
//! exponentially large unfolding), not a deadline. The bound therefore has to
//! survive an instrumented or loaded run: AddressSanitizer, LeakSanitizer and
//! ThreadSanitizer builds are routinely 10-30x slower, and the test threads
//! of one binary share the runner's cores.
//!
//! [`limit`] scales a base bound by [`scale`]. The scale is, in order:
//!
//! 1. `ALKAHEST_TEST_TIME_SCALE` (a positive integer), when set;
//! 2. 20 when any sanitizer runtime option variable (`ASAN_OPTIONS`,
//!    `LSAN_OPTIONS`, `TSAN_OPTIONS`, `MSAN_OPTIONS`, `UBSAN_OPTIONS`) is set,
//!    which every sanitizer shard in CI does;
//! 3. 1 otherwise.
//!
//! A bound that guards "a budget *does* trip" belongs next to a *generous*
//! base (tens of times the budget), scaled the same way; a budget that trips
//! sooner under load is not a failure of that test.

use std::time::Duration;

/// The multiplier applied by [`limit`].
pub(crate) fn scale() -> u32 {
    if let Some(n) = std::env::var("ALKAHEST_TEST_TIME_SCALE")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|n| *n > 0)
    {
        return n;
    }
    const SANITIZER_VARS: [&str; 5] = [
        "ASAN_OPTIONS",
        "LSAN_OPTIONS",
        "TSAN_OPTIONS",
        "MSAN_OPTIONS",
        "UBSAN_OPTIONS",
    ];
    if SANITIZER_VARS.iter().any(|v| std::env::var_os(v).is_some()) {
        20
    } else {
        1
    }
}

/// `base`, stretched for instrumented / loaded runs.
pub(crate) fn limit(base: Duration) -> Duration {
    base.saturating_mul(scale())
}

/// [`limit`] of `secs` seconds.
pub(crate) fn limit_secs(secs: u64) -> Duration {
    limit(Duration::from_secs(secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_never_shrinks_the_base() {
        assert!(limit_secs(2) >= Duration::from_secs(2));
        assert!(scale() >= 1);
    }
}
