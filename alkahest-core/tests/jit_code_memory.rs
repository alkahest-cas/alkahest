//! Dropping a compiled function must give its machine code back.
//!
//! `cranelift_jit`'s `JITModule` leaks its code pages on drop unless
//! `free_memory` is called, and nothing called it: every compile-and-drop kept
//! ~4 kB resident for the life of the process (20 000 compiles: +80 MB). A
//! long-running process that compiles per request — a notebook kernel, a
//! fitting loop recompiling each candidate — grew without bound.
//!
//! This is its own test binary, with a single test, so no concurrently running
//! test can move the resident-set size it measures.

#[cfg(all(target_os = "linux", feature = "cranelift"))]
#[test]
fn compiling_and_dropping_does_not_grow_the_process() {
    use alkahest_cas::kernel::{Domain, ExprPool};
    use alkahest_cas::{compile_with, CompileConfig, CompileTier};

    fn status_kb(field: &str) -> u64 {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        status
            .lines()
            .find(|l| l.starts_with(field))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap()
    }
    // ASan/TSan/MSan reserve terabytes of shadow address space, and ASan
    // quarantines freed memory instead of reusing it, so a resident-set bound
    // means nothing there; the unit test in `cranelift_backend` observes the
    // release path directly.
    if status_kb("VmSize:") > 1 << 30 {
        eprintln!("skipped: sanitizer build (freed memory is quarantined)");
        return;
    }

    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let expr = pool.func("sin", vec![pool.add(vec![x, pool.integer(1_i32)])]);
    let config = CompileConfig {
        expected_evals: None,
        force_tier: Some(CompileTier::Cranelift),
    };
    let compile_and_drop = || {
        let f = compile_with(expr, &[x], &pool, config).unwrap();
        assert_eq!(f.compile_tier(), CompileTier::Cranelift);
        assert_eq!(f.call(&[0.5]), 1.5_f64.sin());
    };

    // Warm up the allocator and the compiler's own caches first.
    for _ in 0..200 {
        compile_and_drop();
    }
    const COMPILES: u64 = 4000;
    let before = status_kb("VmRSS:");
    for _ in 0..COMPILES {
        compile_and_drop();
    }
    let grown = status_kb("VmRSS:").saturating_sub(before);
    eprintln!("{COMPILES} compiles: RSS +{grown} kB");
    // Leaking, this is ~16 MB; freed, it is allocator noise.
    assert!(
        grown < 4 * 1024,
        "{COMPILES} Cranelift compile-and-drops grew RSS by {grown} kB: the JIT code \
         pages of a dropped function are not being freed"
    );
}
