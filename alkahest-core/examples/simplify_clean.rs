//! Discriminator for the simplify_par scaling plateau.
//!
//! `wide_clean` has the same shape as `wide_tree` (one wide Add over N
//! independent children) but the children are already canonical: the rule
//! loop matches and fires nothing, so there is no interning of new nodes and
//! no derivation-log growth — only traversal, rule matching, and scheduling.
//!
//! If this scales near-linearly, the plateau in `wide_tree` comes from
//! intern/allocation contention. If it plateaus too, the limit is in the
//! traversal/scheduling path itself.

use alkahest_cas::kernel::{Domain, ExprId, ExprPool};
use alkahest_cas::simplify::parallel::simplify_par;
use alkahest_cas::simplify::simplify;
use std::time::{Duration, Instant};

/// Already-simplified child: sin(x_i) nested `depth` times, nothing to rewrite.
fn clean_child(pool: &ExprPool, i: usize, depth: usize) -> ExprId {
    let mut e = pool.symbol(format!("x{i}"), Domain::Real);
    for _ in 0..depth {
        e = pool.func("sin", vec![e]);
    }
    e
}

fn wide_clean(pool: &ExprPool, n: usize, depth: usize) -> ExprId {
    let args: Vec<ExprId> = (0..n).map(|i| clean_child(pool, i, depth)).collect();
    pool.add(args)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn time_min(reps: usize, mut f: impl FnMut() -> ExprId) -> Duration {
    let mut best = Duration::from_secs(u64::MAX);
    for _ in 0..reps {
        let t = Instant::now();
        let v = f();
        std::hint::black_box(v);
        let d = t.elapsed();
        if d < best {
            best = d;
        }
    }
    best
}

fn main() {
    let reps = 5;
    let threads = [1usize, 2, 4, 8, 16, 32];

    for &(n, d) in &[(1024usize, 8usize), (4096, 8)] {
        let build = |p: &ExprPool| wide_clean(p, n, d);
        let build_t = time_min(reps, || {
            let pool = ExprPool::new();
            build(&pool)
        });

        // work check: no rewrite steps should fire
        let pool = ExprPool::new();
        let e = build(&pool);
        let steps = simplify(e, &pool).log.len();

        let seq_t = time_min(reps, || {
            let pool = ExprPool::new();
            let e = build(&pool);
            simplify(e, &pool).value
        })
        .saturating_sub(build_t);

        println!("\n=== wide_clean n={n} depth={d}  (seq rewrite steps = {steps}) ===");
        println!("  build          {:>9.2} ms", ms(build_t));
        println!("  simplify (seq) {:>9.2} ms", ms(seq_t));

        let mut base = None;
        for &nt in &threads {
            let tp = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            let t = time_min(reps, || {
                let pool = ExprPool::new();
                let e = build(&pool);
                tp.install(|| simplify_par(e, &pool).value)
            })
            .saturating_sub(build_t);
            if base.is_none() {
                base = Some(t);
            }
            println!(
                "  par t={nt:<3}      {:>9.2} ms   scale-vs-1t {:>5.2}x   vs-seq {:>5.2}x",
                ms(t),
                base.unwrap().as_secs_f64() / t.as_secs_f64(),
                seq_t.as_secs_f64() / t.as_secs_f64()
            );
        }
    }
}
