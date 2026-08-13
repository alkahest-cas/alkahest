//! Re-export PyO3's build-time cfgs so this crate can gate on them.
//!
//! `pyo3::buffer` is gated, in PyO3 itself, as
//! `#![cfg(any(not(Py_LIMITED_API), Py_3_11))]` — the buffer protocol is not
//! part of the stable ABI before 3.11. Without these cfgs a build against a
//! limited-API 3.9/3.10 interpreter fails with `unresolved import pyo3::buffer`
//! *only in the release wheel matrix*, which is the one place nobody looks
//! until tagging.
fn main() {
    // Declare the cfgs before setting them: rustc ≥ 1.80 warns on unknown cfg
    // names, and this workspace builds with `-D warnings`.
    println!("cargo:rustc-check-cfg=cfg(Py_LIMITED_API)");
    println!("cargo:rustc-check-cfg=cfg(Py_3_11)");
    pyo3_build_config::use_pyo3_cfgs();
}
