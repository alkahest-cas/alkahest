//! Sequential vs fork-join (`simplify_par`) vs level-scheduled (`simplify_redex`).
//!
//! Prints wall time per scheduler across thread counts for six expression
//! shapes, after checking that all three agree on the result.
//!
//!   cargo run --release --features parallel --example simplify_three_way

use alkahest_cas::kernel::{Domain, ExprId, ExprPool};
use alkahest_cas::simplify::parallel::simplify_par;
use alkahest_cas::simplify::redex::simplify_redex;
use alkahest_cas::simplify::simplify;
use std::time::{Duration, Instant};

fn junk(pool: &ExprPool, x: ExprId, depth: usize) -> ExprId {
    let one = pool.integer(1);
    let zero = pool.integer(0);
    let mut e = x;
    for _ in 0..depth {
        e = pool.mul(vec![e, one]);
        e = pool.add(vec![e, zero]);
        e = pool.pow(e, one);
    }
    e
}

/// Wide `Add` of independent children — the shape `simplify_par` was built for.
fn wide_tree(pool: &ExprPool, n: usize, depth: usize) -> ExprId {
    let args: Vec<ExprId> = (0..n)
        .map(|i| {
            let x = pool.symbol(format!("x{i}"), Domain::Real);
            junk(pool, x, depth)
        })
        .collect();
    pool.add(args)
}

/// Wide `Add` over one shared subexpression — a DAG, the common CAS shape.
fn wide_shared(pool: &ExprPool, n: usize, depth: usize) -> ExprId {
    let x = pool.symbol("x", Domain::Real);
    let shared = junk(pool, x, depth);
    let args: Vec<ExprId> = (0..n)
        .map(|i| {
            let c = pool.integer(i as i64 + 1);
            pool.mul(vec![shared, c])
        })
        .collect();
    pool.add(args)
}

/// Wide `Add` of already-canonical children: traversal cost with no rewrites.
fn wide_clean(pool: &ExprPool, n: usize, depth: usize) -> ExprId {
    let args: Vec<ExprId> = (0..n)
        .map(|i| {
            let mut e = pool.symbol(format!("x{i}"), Domain::Real);
            for _ in 0..depth {
                e = pool.func("sin", vec![e]);
            }
            e
        })
        .collect();
    pool.add(args)
}

/// A single deep chain: one node per level, no width to exploit.
fn deep(pool: &ExprPool, depth: usize) -> ExprId {
    let x = pool.symbol("x", Domain::Real);
    junk(pool, x, depth)
}

/// Many medium chains — deep *and* wide, where level scheduling should win.
fn forest(pool: &ExprPool, n: usize, depth: usize) -> ExprId {
    let args: Vec<ExprId> = (0..n)
        .map(|i| {
            let x = pool.symbol(format!("x{i}"), Domain::Real);
            let e = junk(pool, x, depth);
            pool.func("sin", vec![e])
        })
        .collect();
    pool.add(args)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Time `op` alone: the expression is rebuilt in a fresh pool before each rep
/// (hash-consing makes a second run over the same pool much cheaper), and the
/// clock only covers the call itself — no subtracting a separately measured
/// build, which saturates to zero when the op is faster than the build.
fn time_op(
    reps: usize,
    build: &dyn Fn(&ExprPool) -> ExprId,
    op: &dyn Fn(ExprId, &ExprPool) -> ExprId,
) -> Duration {
    let mut best = Duration::from_secs(u64::MAX);
    for _ in 0..reps {
        let pool = ExprPool::new();
        let e = build(&pool);
        let t = Instant::now();
        std::hint::black_box(op(e, &pool));
        let d = t.elapsed();
        if d < best {
            best = d;
        }
    }
    best
}

fn run(name: &str, build: &dyn Fn(&ExprPool) -> ExprId, threads: &[usize], reps: usize) {
    let seq = time_op(reps, build, &|e, pool| simplify(e, pool).value);

    // Agreement check before timing anything else.
    let pool = ExprPool::new();
    let e = build(&pool);
    let (a, b, c) = (
        simplify(e, &pool).value,
        simplify_par(e, &pool).value,
        simplify_redex(e, &pool).value,
    );
    let agree = a == b && b == c;

    println!(
        "\n=== {name} ===   agreement: {}",
        if agree { "ok" } else { "MISMATCH" }
    );
    println!("  {:<10} {:>9.2} ms", "sequential", ms(seq));
    println!(
        "  {:<10} {:>9} {:>9}   (speedup vs sequential)",
        "threads", "fork-join", "level"
    );

    for &nt in threads {
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap();
        let par = time_op(reps, build, &|e, pool| {
            tp.install(|| simplify_par(e, pool).value)
        });
        let lvl = time_op(reps, build, &|e, pool| {
            tp.install(|| simplify_redex(e, pool).value)
        });
        println!(
            "  t={:<8} {:>7.2}ms {:>7.2}ms   fork-join {:.2}x  level {:.2}x",
            nt,
            ms(par),
            ms(lvl),
            seq.as_secs_f64() / par.as_secs_f64(),
            seq.as_secs_f64() / lvl.as_secs_f64(),
        );
    }
}

fn main() {
    let threads: Vec<usize> = std::env::var("THREADS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32]);
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    println!(
        "cores={}  reps={reps}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );

    run(
        "wide_tree n=1024 d=8",
        &|p| wide_tree(p, 1024, 8),
        &threads,
        reps,
    );
    run(
        "wide_tree n=4096 d=4",
        &|p| wide_tree(p, 4096, 4),
        &threads,
        reps,
    );
    run(
        "wide_shared n=1024 d=8",
        &|p| wide_shared(p, 1024, 8),
        &threads,
        reps,
    );
    run(
        "wide_clean n=4096 d=8",
        &|p| wide_clean(p, 4096, 8),
        &threads,
        reps,
    );
    run(
        "forest n=1024 d=32",
        &|p| forest(p, 1024, 32),
        &threads,
        reps,
    );
    run("deep d=2000", &|p| deep(p, 2000), &threads, reps);
}
