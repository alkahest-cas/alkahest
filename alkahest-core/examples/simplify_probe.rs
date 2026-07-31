//! Follow-up probes for the simplify_par scaling study:
//!   1. work done (derivation-log length) seq vs par — measures memo effect
//!   2. deep-chain recursion: caller thread vs rayon worker thread
//!   3. `expand: true` ruleset parity between seq and par

use alkahest_cas::kernel::{Domain, ExprId, ExprPool};
use alkahest_cas::simplify::parallel::simplify_par;
use alkahest_cas::simplify::{simplify, simplify_with, SimplifyConfig};

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

fn wide_tree(pool: &ExprPool, n: usize, depth: usize) -> ExprId {
    let args: Vec<ExprId> = (0..n)
        .map(|i| {
            let x = pool.symbol(format!("x{i}"), Domain::Real);
            junk(pool, x, depth)
        })
        .collect();
    pool.add(args)
}

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

fn probe_work(name: &str, build: impl Fn(&ExprPool) -> ExprId) {
    let pool = ExprPool::new();
    let e = build(&pool);
    let n_before = pool.len();
    let s = simplify(e, &pool);
    let seq_steps = s.log.len();
    let seq_nodes = pool.len() - n_before;

    let pool2 = ExprPool::new();
    let e2 = build(&pool2);
    let n_before2 = pool2.len();
    let p = simplify_par(e2, &pool2);
    let par_steps = p.log.len();
    let par_nodes = pool2.len() - n_before2;

    println!(
        "{name:28} seq_steps={seq_steps:>8}  par_steps={par_steps:>8}  ratio={:>6.1}x   \
         seq_new_nodes={seq_nodes:>6} par_new_nodes={par_nodes:>6}",
        par_steps as f64 / seq_steps.max(1) as f64
    );
}

fn probe_deep_stack() {
    println!("\n--- deep chain: caller thread vs rayon worker ---");
    for depth in [500usize, 1000, 2000, 4000] {
        // (a) directly on this (main) thread, 8 MiB stack
        let r = std::panic::catch_unwind(|| {
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let e = junk(&pool, x, depth);
            simplify_par(e, &pool).value
        });
        println!(
            "depth={depth:<5} main-thread simplify_par: {}",
            if r.is_ok() { "ok" } else { "PANIC" }
        );
    }
    println!("(rayon-worker variant is run separately — it aborts the process on overflow)");
}

fn probe_deep_on_worker(depth: usize) {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let ok = tp.install(|| {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = junk(&pool, x, depth);
        simplify_par(e, &pool).value
    });
    println!("depth={depth} on rayon worker: ok ({ok:?})");
}

fn probe_expand_parity() {
    println!("\n--- expand: true ruleset parity ---");
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let two = pool.integer(2);
    let sum = pool.add(vec![x, y]);
    let sq = pool.pow(sum, two); // (x + y)^2
    let cfg = SimplifyConfig {
        expand: true,
        ..Default::default()
    };
    let seq = simplify_with(
        sq,
        &pool,
        &alkahest_cas::simplify::rules_for_config(&cfg),
        cfg.clone(),
    )
    .value;
    let par = alkahest_cas::simplify::parallel::simplify_par_with_config(sq, &pool, &cfg).value;
    println!(
        "(x+y)^2  expand=true\n  seq = {}\n  par = {}\n  equal: {}",
        alkahest_cas::kernel::display::render_unicode(seq, &pool),
        alkahest_cas::kernel::display::render_unicode(par, &pool),
        seq == par
    );
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    if mode == "worker" {
        let depth: usize = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(2000);
        probe_deep_on_worker(depth);
        return;
    }

    println!("--- work done (derivation-log steps) ---");
    probe_work("wide_tree n=256 d=8", |p| wide_tree(p, 256, 8));
    probe_work("wide_tree n=1024 d=8", |p| wide_tree(p, 1024, 8));
    probe_work("wide_shared n=256 d=8", |p| wide_shared(p, 256, 8));
    probe_work("wide_shared n=1024 d=8", |p| wide_shared(p, 1024, 8));

    probe_deep_stack();
    probe_expand_parity();
}
