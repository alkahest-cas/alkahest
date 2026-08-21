//! Memory accounting for the exact-arithmetic paths, and the address-space
//! guard that turns an imminent out-of-memory `abort()` into a refusal.
//!
//! # The failure this exists to remove
//!
//! `rug`/GMP's default reaction to a failed allocation is to print
//! `GNU MP: Cannot allocate memory (size=N)` and call `abort()`. The Rust
//! allocator's is `memory allocation of N bytes failed` followed by the same
//! `abort()`. Neither is catchable: an exact-rational solve that outgrows the
//! machine takes the whole interpreter with it, and an unattended research
//! loop loses every result it was holding, not just the offending call.
//!
//! # What is enforceable, and what is not
//!
//! GMP's contract for a replacement allocation function is that it *must not
//! return `NULL`* — the library has no failure path to take. Nor may a
//! replacement unwind: a Rust `panic!` crossing a C frame is undefined
//! behaviour (and, since Rust 1.81, aborts anyway). So a custom allocator
//! cannot itself convert an out-of-memory condition into an error.
//!
//! What it *can* do soundly is **count**. The functions installed by
//! [`install`] delegate to whatever GMP was already using and maintain a
//! process-wide live-byte total. Nothing in them allocates, unwinds, or
//! returns `NULL`, so they are safe to run inside GMP frames. The refusal
//! then happens at Alkahest's own cooperative checkpoints
//! ([`crate::budget::check_all`]), which are ordinary Rust code returning an
//! ordinary `Err` — a *pre-flight* refusal, before the allocation that would
//! have died is attempted.
//!
//! Two ceilings feed those checkpoints:
//!
//! * **`Budget::max_bytes`** — an explicit, caller-supplied ceiling on how
//!   much GMP memory one guarded block may hold live. See
//!   [`crate::budget::enter_with_memory`].
//! * **The address-space guard** — active with no budget at all. When the
//!   process runs under a finite `RLIMIT_AS` (`ulimit -v`, a container limit,
//!   a batch scheduler), [`headroom_exhausted`] reports when the process has
//!   climbed to within [`reserve_bytes`] of that limit, and the checkpoint
//!   refuses. This is what makes the *default-arguments* case survivable:
//!   the operator already said how much the process may have, so Alkahest
//!   stops inside that number instead of dying at it. With no limit set
//!   (`ulimit -v unlimited`), the guard is inert and behaviour is unchanged.
//!
//! # What remains
//!
//! The guard is checkpoint-granular. A single allocation large enough to jump
//! the whole reserve in one step, between two consecutive checkpoints, still
//! aborts — nothing short of a fallible allocator can fix that, and GMP does
//! not have one. `reserve_bytes` is sized to make that unlikely rather than
//! impossible. Address-space *usage* is only observable on Linux
//! (`/proc/self/statm`); on other platforms the guard degrades to the
//! `Budget::max_bytes` ceiling alone.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Once, OnceLock};

use gmp_mpfr_sys::gmp;

// ---------------------------------------------------------------------------
// GMP allocation accounting
// ---------------------------------------------------------------------------

/// Live bytes currently held by GMP allocations, process-wide.
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

/// The allocation functions GMP was using before [`install`] wrapped them,
/// stored as raw addresses because a function pointer is not a `const`-
/// initialisable atomic. Written once, before `INSTALLED` is set.
static ORIG_ALLOC: AtomicUsize = AtomicUsize::new(0);
static ORIG_REALLOC: AtomicUsize = AtomicUsize::new(0);
static ORIG_FREE: AtomicUsize = AtomicUsize::new(0);

static INSTALLED: AtomicBool = AtomicBool::new(false);
static INSTALL_ONCE: Once = Once::new();

type AllocFn = extern "C" fn(usize) -> *mut c_void;
type ReallocFn = extern "C" fn(*mut c_void, usize, usize) -> *mut c_void;
type FreeFn = unsafe extern "C" fn(*mut c_void, usize);

fn add_live(n: usize) {
    LIVE_BYTES.fetch_add(n as u64, Ordering::Relaxed);
}

fn sub_live(n: usize) {
    // Saturating, not wrapping: blocks allocated *before* `install` ran are
    // freed through our wrapper without ever having been counted, so the
    // total can legitimately try to go negative. Under-counting only ever
    // makes the ceiling fire late, never spuriously.
    let _ = LIVE_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_sub(n as u64))
    });
}

extern "C" fn wrap_alloc(size: usize) -> *mut c_void {
    let orig: AllocFn = unsafe { std::mem::transmute(ORIG_ALLOC.load(Ordering::Acquire)) };
    let p = orig(size);
    add_live(size);
    p
}

extern "C" fn wrap_realloc(ptr: *mut c_void, old: usize, new: usize) -> *mut c_void {
    let orig: ReallocFn = unsafe { std::mem::transmute(ORIG_REALLOC.load(Ordering::Acquire)) };
    let p = orig(ptr, old, new);
    sub_live(old);
    add_live(new);
    p
}

unsafe extern "C" fn wrap_free(ptr: *mut c_void, size: usize) {
    let orig: FreeFn = std::mem::transmute(ORIG_FREE.load(Ordering::Acquire));
    orig(ptr, size);
    sub_live(size);
}

/// Install the counting wrappers around GMP's current allocation functions.
///
/// Idempotent and thread-safe (guarded by a [`Once`]); returns `true` if
/// accounting is active. Call it as early as the embedding allows — the
/// Python extension does it from its module initialiser — but it is *not*
/// required to run before GMP's first allocation: the wrappers delegate to
/// the functions that were installed at the time, so a block allocated
/// before installation is still freed by the allocator that produced it, and
/// the only consequence of installing late is that the live total starts from
/// a baseline it never saw allocated (handled by [`sub_live`]'s saturation).
///
/// GMP's own documentation warns that changing the allocation functions while
/// other threads are inside GMP is unsafe; the [`Once`] makes the swap happen
/// exactly once, and the intended call site is process start-up.
pub fn install() -> bool {
    INSTALL_ONCE.call_once(|| {
        let mut alloc: gmp::allocate_function = None;
        let mut realloc: gmp::reallocate_function = None;
        let mut free: gmp::free_function = None;
        // SAFETY: three out-pointers to live locals, which is exactly what
        // `mp_get_memory_functions` expects.
        unsafe { gmp::get_memory_functions(&mut alloc, &mut realloc, &mut free) };
        let (Some(alloc), Some(realloc), Some(free)) = (alloc, realloc, free) else {
            // GMP always reports concrete functions (the defaults if nothing
            // was installed); if it somehow does not, leave it alone.
            return;
        };
        ORIG_ALLOC.store(alloc as usize, Ordering::Release);
        ORIG_REALLOC.store(realloc as usize, Ordering::Release);
        ORIG_FREE.store(free as usize, Ordering::Release);
        // SAFETY: the wrappers delegate to the functions just captured, never
        // return NULL where the original did not, never unwind, and never
        // allocate — the three things GMP requires of a replacement.
        unsafe {
            gmp::set_memory_functions(Some(wrap_alloc), Some(wrap_realloc), Some(wrap_free));
        }
        INSTALLED.store(true, Ordering::Release);
    });
    INSTALLED.load(Ordering::Acquire)
}

/// `true` if [`install`] has swapped in the counting wrappers.
pub fn is_installed() -> bool {
    INSTALLED.load(Ordering::Acquire)
}

/// Bytes currently held live by GMP allocations, process-wide.
///
/// Zero when [`install`] has not run. Process-wide rather than per-thread
/// because GMP's allocation hooks are global: a block allocated on one thread
/// may be freed on another, so a thread-local total could not stay balanced.
pub fn gmp_live_bytes() -> u64 {
    LIVE_BYTES.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Address-space guard
// ---------------------------------------------------------------------------

/// The soft `RLIMIT_AS` of this process in bytes, or `None` when it is
/// unlimited (or cannot be read).
///
/// Read once and cached: a process that raises its own limit mid-run is not a
/// case worth a syscall on every checkpoint, and caching can only make the
/// guard *more* conservative.
pub fn address_space_limit() -> Option<u64> {
    static LIMIT: OnceLock<Option<u64>> = OnceLock::new();
    *LIMIT.get_or_init(read_address_space_limit)
}

#[cfg(unix)]
fn read_address_space_limit() -> Option<u64> {
    let mut rl = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `getrlimit` writes a `struct rlimit` through the out-pointer.
    if unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut rl) } != 0 {
        return None;
    }
    if rl.rlim_cur == libc::RLIM_INFINITY {
        None
    } else {
        // The cast is a no-op where `rlim_t` is already `u64` (Linux, macOS)
        // and load-bearing where it is not, so it stays.
        #[allow(clippy::unnecessary_cast)]
        Some(rl.rlim_cur as u64)
    }
}

#[cfg(not(unix))]
fn read_address_space_limit() -> Option<u64> {
    None
}

/// Virtual address space currently mapped by this process, in bytes.
///
/// Linux only (`/proc/self/statm`); `None` elsewhere, which disables the
/// address-space guard rather than guessing.
#[cfg(target_os = "linux")]
pub fn address_space_used() -> Option<u64> {
    use std::io::Read;
    let mut buf = [0u8; 64];
    let mut f = std::fs::File::open("/proc/self/statm").ok()?;
    let n = f.read(&mut buf).ok()?;
    let text = std::str::from_utf8(buf.get(..n)?).ok()?;
    let pages: u64 = text.split_whitespace().next()?.parse().ok()?;
    Some(pages.saturating_mul(page_size()))
}

#[cfg(not(target_os = "linux"))]
pub fn address_space_used() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn page_size() -> u64 {
    static PAGE: OnceLock<u64> = OnceLock::new();
    *PAGE.get_or_init(|| {
        // SAFETY: `sysconf` takes an int and returns a long; no pointers.
        let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if n > 0 {
            n as u64
        } else {
            4096
        }
    })
}

/// Headroom the address-space guard keeps in reserve below `RLIMIT_AS`.
///
/// Sized to be crossed by *many* checkpoint intervals, not by one: the
/// allocations between two consecutive checkpoints in an exact-rational
/// elimination are limb arrays measured in kilobytes, so 32 MiB is hundreds of
/// them.
///
/// It is deliberately **not** a large fraction of the limit. `RLIMIT_AS` caps
/// virtual address space, and importing the extension already maps ~600 MB of
/// it (arena reservations and thread stacks, ~46 MB of which is resident), so
/// a reserve of "an eighth of the limit" would refuse every call under a
/// 900 MB `ulimit -v` — including the ones that fit comfortably today. A flat
/// floor with a gentle fraction for large limits keeps the guard out of the
/// way until the process is genuinely at the edge.
pub fn reserve_bytes(limit: u64) -> u64 {
    const FLOOR: u64 = 32 * 1024 * 1024;
    const CEILING: u64 = 256 * 1024 * 1024;
    (limit / 64).clamp(FLOOR, CEILING)
}

/// `Some((used, limit))` when the process has climbed to within
/// [`reserve_bytes`] of its address-space limit, else `None`.
///
/// `None` — the guard is inert — when no finite `RLIMIT_AS` is set, or when
/// address-space usage is not observable on this platform.
pub fn headroom_exhausted() -> Option<(u64, u64)> {
    let limit = address_space_limit()?;
    let used = address_space_used()?;
    if used.saturating_add(reserve_bytes(limit)) >= limit {
        Some((used, limit))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gmp_accounting_tracks_a_big_rational() {
        assert!(install(), "GMP accounting must install");
        let before = gmp_live_bytes();
        let big = {
            let mut z = rug::Integer::from(1);
            z <<= 8_000_000; // ~1 MB of limbs
            z
        };
        let during = gmp_live_bytes();
        assert!(
            during > before + 500_000,
            "expected the shift to be counted: {before} -> {during}"
        );
        drop(big);
        let after = gmp_live_bytes();
        assert!(
            after < during,
            "freeing must decrement the live total: {during} -> {after}"
        );
    }

    #[test]
    fn reserve_is_clamped_to_the_floor_and_ceiling() {
        assert_eq!(reserve_bytes(1024), 32 * 1024 * 1024);
        assert_eq!(reserve_bytes(900_000_000), 32 * 1024 * 1024);
        assert_eq!(reserve_bytes(64 * 1024 * 1024 * 1024), 256 * 1024 * 1024);
    }

    #[test]
    fn headroom_guard_is_inert_without_a_limit() {
        // The test binary itself runs with no `ulimit -v` in CI, so the guard
        // must not fire. (When a limit *is* set the subprocess regression
        // test in tests/ exercises the firing path.)
        if address_space_limit().is_none() {
            assert_eq!(headroom_exhausted(), None);
        }
    }
}
