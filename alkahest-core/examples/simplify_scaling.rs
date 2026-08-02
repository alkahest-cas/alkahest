//! Scaling curve for `simplify_par` vs sequential `simplify`.
//!
//! Run with:
//!   cargo run --release --features parallel --example simplify_scaling
//!
//! Three workload shapes:
//!   * `wide_tree`   — Add of N *distinct* children (pure tree, no sharing)
//!   * `wide_shared` — Add of N children over a shared heavy subexpression (DAG)
//!   * `deep`        — one deep chain (no wide Add/Mul, so no par path at all)

use alkahest_cas::kernel::{Domain, ExprId, ExprPool};
use alkahest_cas::simplify::parallel::simplify_par;
use alkahest_cas::simplify::simplify;
use std::time::{Duration, Instant};

/// Wrap `x` in `depth` layers of algebraic junk that the rule engine removes:
/// `((e * 1) + 0)^1`.
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

/// Add of `n` children, each built over its own variable → no shared structure.
fn wide_tree(pool: &ExprPool, n: usize, depth: usize) -> ExprId {
    let args: Vec<ExprId> = (0..n)
        .map(|i| {
            let x = pool.symbol(format!("x{i}"), Domain::Real);
            junk(pool, x, depth)
        })
        .collect();
    pool.add(args)
}

/// Add of `n` children that all sit on top of one shared heavy subexpression.
/// Hash-consing makes this a DAG; the sequential memo collapses it, the
/// parallel path has no memo.
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

/// One deep chain — never hits the wide Add/Mul par path.
fn deep(pool: &ExprPool, depth: usize) -> ExprId {
    let x = pool.symbol("x", Domain::Real);
    junk(pool, x, depth)
}

fn time_min<F: FnMut() -> ExprId>(reps: usize, mut f: F) -> (Duration, ExprId) {
    let mut best = Duration::from_secs(u64::MAX);
    let mut out = None;
    for _ in 0..reps {
        // Fresh pool per rep is handled by the caller's closure.
        let t = Instant::now();
        let v = f();
        let d = t.elapsed();
        if d < best {
            best = d;
        }
        out = Some(v);
    }
    (best, out.unwrap())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn run_case(name: &str, build: &dyn Fn(&ExprPool) -> ExprId, threads: &[usize], reps: usize) {
    // --- sequential baseline (fresh pool per rep) ---
    let (seq_t, seq_v) = time_min(reps, || {
        let pool = ExprPool::new();
        let e = build(&pool);
        let t = Instant::now();
        let r = simplify(e, &pool).value;
        // fold the inner timing out: we time the whole closure, so subtract
        // nothing here — build cost is measured separately below.
        let _ = t;
        r
    });

    // Measure build cost alone so we can subtract it.
    let (build_t, _) = time_min(reps, || {
        let pool = ExprPool::new();
        build(&pool)
    });

    let seq_net = seq_t.saturating_sub(build_t);
    let seq_str = {
        let pool = ExprPool::new();
        let e = build(&pool);
        let r = simplify(e, &pool).value;
        alkahest_cas::kernel::display::render_unicode(r, &pool)
    };

    println!("\n=== {name} ===");
    println!("  build          {:>10.2} ms", ms(build_t));
    println!(
        "  simplify (seq) {:>10.2} ms   (net of build)   result_len={}",
        ms(seq_net),
        seq_str.len()
    );
    let _ = seq_v;

    let mut base_par: Option<Duration> = None;
    for &nt in threads {
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap();
        let (par_t, _) = time_min(reps, || {
            let pool = ExprPool::new();
            let e = build(&pool);
            tp.install(|| simplify_par(e, &pool).value)
        });
        let par_net = par_t.saturating_sub(build_t);
        if base_par.is_none() {
            base_par = Some(par_net);
        }
        let self_speedup = base_par.unwrap().as_secs_f64() / par_net.as_secs_f64();
        let vs_seq = seq_net.as_secs_f64() / par_net.as_secs_f64();
        println!(
            "  par t={nt:<3}      {:>10.2} ms   scale-vs-1t {:>5.2}x   vs-seq {:>5.2}x",
            ms(par_net),
            self_speedup,
            vs_seq
        );
    }

    // Correctness cross-check: par and seq must agree structurally.
    let pool = ExprPool::new();
    let e = build(&pool);
    let a = simplify(e, &pool).value;
    let b = simplify_par(e, &pool).value;
    println!(
        "  agreement: {}",
        if a == b {
            "same ExprId".to_string()
        } else {
            format!(
                "DIFFER seq={} par={}",
                alkahest_cas::kernel::display::render_unicode(a, &pool),
                alkahest_cas::kernel::display::render_unicode(b, &pool)
            )
        }
    );
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

    println!("cores={}  reps={reps}", num_cpus_hint());

    for &(n, d) in &[(256usize, 8usize), (1024, 8), (4096, 4)] {
        run_case(
            &format!("wide_tree n={n} depth={d}"),
            &move |p: &ExprPool| wide_tree(p, n, d),
            &threads,
            reps,
        );
    }

    for &(n, d) in &[(256usize, 8usize), (1024, 8)] {
        run_case(
            &format!("wide_shared n={n} depth={d}"),
            &move |p: &ExprPool| wide_shared(p, n, d),
            &threads,
            reps,
        );
    }

    run_case(
        "deep depth=2000",
        &|p: &ExprPool| deep(p, 2000),
        &threads,
        reps,
    );
}

fn num_cpus_hint() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}
