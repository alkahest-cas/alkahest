//! Isolates the per-node read cost in the parallel traversal.
//!
//! `simplify_node_par` starts every node visit with `let data = pool.get(expr)`
//! which *clones* the `ExprData` (a `Vec<ExprId>` for Add/Mul, a `String` for
//! Func names) — one heap allocation per node visit, on every worker.
//! `ExprPool::with` borrows the same node with no allocation.
//!
//! This measures both under identical parallel traversal.

use alkahest_cas::kernel::{Domain, ExprData, ExprId, ExprPool};
use rayon::prelude::*;
use std::time::{Duration, Instant};

fn build(pool: &ExprPool, n: usize, depth: usize) -> Vec<ExprId> {
    (0..n)
        .map(|i| {
            let mut e = pool.symbol(format!("x{i}"), Domain::Real);
            for _ in 0..depth {
                e = pool.func("sin", vec![e]);
            }
            e
        })
        .collect()
}

/// Walk the tree rooted at `id`, cloning each node (mirrors `pool.get`).
fn walk_clone(pool: &ExprPool, id: ExprId) -> usize {
    let data = pool.get(id);
    let mut n = 1;
    match data {
        ExprData::Add(args) | ExprData::Mul(args) => {
            for a in args {
                n += walk_clone(pool, a);
            }
        }
        ExprData::Func { args, .. } => {
            for a in args {
                n += walk_clone(pool, a);
            }
        }
        ExprData::Pow { base, exp } => {
            n += walk_clone(pool, base) + walk_clone(pool, exp);
        }
        _ => {}
    }
    n
}

/// Same walk via the borrowing API — no per-node allocation.
fn walk_borrow(pool: &ExprPool, id: ExprId) -> usize {
    pool.with(id, |data| {
        let mut n = 1;
        match data {
            ExprData::Add(args) | ExprData::Mul(args) => {
                for &a in args {
                    n += walk_borrow(pool, a);
                }
            }
            ExprData::Func { args, .. } => {
                for &a in args {
                    n += walk_borrow(pool, a);
                }
            }
            ExprData::Pow { base, exp } => {
                n += walk_borrow(pool, *base) + walk_borrow(pool, *exp);
            }
            _ => {}
        }
        n
    })
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let n = 4096;
    let depth = 32;
    let pool = ExprPool::new();
    let roots = build(&pool, n, depth);
    println!("n={n} depth={depth}  pool_nodes={}", pool.len());

    for (label, f) in [
        ("get (clone)", walk_clone as fn(&ExprPool, ExprId) -> usize),
        (
            "with (borrow)",
            walk_borrow as fn(&ExprPool, ExprId) -> usize,
        ),
    ] {
        println!("\n--- {label} ---");
        let mut base: Option<Duration> = None;
        for nt in [1usize, 2, 4, 8, 16, 32] {
            let tp = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            let mut best = Duration::from_secs(u64::MAX);
            for _ in 0..5 {
                let t = Instant::now();
                let total: usize = tp.install(|| roots.par_iter().map(|&r| f(&pool, r)).sum());
                let d = t.elapsed();
                std::hint::black_box(total);
                if d < best {
                    best = d;
                }
            }
            if base.is_none() {
                base = Some(best);
            }
            println!(
                "  t={nt:<3} {:>8.2} ms   scale {:>5.2}x",
                ms(best),
                base.unwrap().as_secs_f64() / best.as_secs_f64()
            );
        }
    }
}
