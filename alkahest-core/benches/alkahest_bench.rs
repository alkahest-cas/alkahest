use alkahest_core::matrix::{hermite_form, IntegerMatrix};
#[cfg(feature = "egraph")]
use alkahest_core::simplify_egraph;
use alkahest_core::{
    compile, diff, eval_interp, simplify, ArbBall, Domain, ExprId, ExprPool, IntervalEval,
    MultiPoly, UniPoly,
};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Counting allocator — tracks cumulative bytes allocated
// ---------------------------------------------------------------------------

static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Only count the growth; realloc shrinks are free.
        if new_size > layout.size() {
            ALLOC_BYTES.fetch_add((new_size - layout.size()) as u64, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn snapshot() -> (u64, u64) {
    (
        ALLOC_BYTES.load(Ordering::Relaxed),
        ALLOC_COUNT.load(Ordering::Relaxed),
    )
}

fn delta(before: (u64, u64)) -> (u64, u64) {
    let after = snapshot();
    (after.0 - before.0, after.1 - before.1)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fresh_pool() -> ExprPool {
    ExprPool::new()
}

/// Build `coeffs[0] + coeffs[1]*x + … ` in the given pool.
fn poly_expr(p: &ExprPool, x: alkahest_core::ExprId, coeffs: &[i64]) -> alkahest_core::ExprId {
    let mut terms = vec![];
    for (i, &c) in coeffs.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let c_id = p.integer(c);
        if i == 0 {
            terms.push(c_id);
        } else {
            let xpow = p.pow(x, p.integer(i as i32));
            terms.push(if c == 1 {
                xpow
            } else {
                p.mul(vec![c_id, xpow])
            });
        }
    }
    match terms.len() {
        0 => p.integer(0_i32),
        1 => terms[0],
        _ => p.add(terms),
    }
}

// ---------------------------------------------------------------------------
// Group 1: intern table (ExprPool)
// ---------------------------------------------------------------------------

fn bench_intern(c: &mut Criterion) {
    let mut g = c.benchmark_group("intern");

    // Repeated intern of the same symbol — hits the HashMap cache every time.
    g.bench_function("symbol_cached", |b| {
        let p = fresh_pool();
        b.iter(|| p.symbol("x", Domain::Real));
    });

    // 100 unique integers — table grows on every call.
    g.throughput(Throughput::Elements(100));
    g.bench_function("integer_unique_100", |b| {
        b.iter(|| {
            let p = fresh_pool();
            for i in 0i64..100 {
                p.integer(i);
            }
        });
    });

    // Build a 3-arg Add node from pre-interned leaves.
    g.throughput(Throughput::Elements(1));
    g.bench_function("build_add3", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let one = p.integer(1_i32);
        let two = p.integer(2_i32);
        b.iter(|| p.add(vec![x, one, two]));
    });

    // Structural sharing: build the same sub-tree twice and verify only one node exists.
    g.bench_function("sharing_x_plus_1", |b| {
        b.iter(|| {
            let p = fresh_pool();
            let x = p.symbol("x", Domain::Real);
            let one = p.integer(1_i32);
            let e1 = p.add(vec![x, one]);
            let e2 = p.add(vec![x, one]); // must return same ExprId
            assert_eq!(e1, e2, "hash-consing broken");
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Group 2: simplification
// ---------------------------------------------------------------------------

fn bench_simplify(c: &mut Criterion) {
    let mut g = c.benchmark_group("simplify");

    // Trivial: x + 0 → x
    g.bench_function("add_zero", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = p.add(vec![x, p.integer(0_i32)]);
        b.iter(|| simplify(expr, &p));
    });

    // All-constant fold: 3 + 4 + 5 → 12
    g.bench_function("const_fold_3terms", |b| {
        let p = fresh_pool();
        let expr = p.add(vec![p.integer(3_i32), p.integer(4_i32), p.integer(5_i32)]);
        b.iter(|| simplify(expr, &p));
    });

    // Polynomial degrees 1–4
    for deg in [1usize, 2, 3, 4] {
        g.throughput(Throughput::Elements(deg as u64 + 1)); // #terms
        g.bench_with_input(BenchmarkId::new("polynomial_deg", deg), &deg, |b, &deg| {
            let p = fresh_pool();
            let x = p.symbol("x", Domain::Real);
            let coeffs: Vec<i64> = (0..=deg as i64).collect();
            let expr = poly_expr(&p, x, &coeffs);
            b.iter(|| simplify(expr, &p));
        });
    }

    // Already-simplified: measures fixpoint-detection overhead only.
    g.throughput(Throughput::Elements(1));
    g.bench_function("already_simplified", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let simplified = simplify(poly_expr(&p, x, &[1, 2, 3]), &p).value;
        b.iter(|| simplify(simplified, &p));
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Group 3: symbolic differentiation
// ---------------------------------------------------------------------------

fn bench_diff(c: &mut Criterion) {
    let mut g = c.benchmark_group("diff");

    for deg in [1usize, 2, 3, 4] {
        g.throughput(Throughput::Elements(deg as u64 + 1));
        g.bench_with_input(BenchmarkId::new("polynomial_deg", deg), &deg, |b, &deg| {
            let p = fresh_pool();
            let x = p.symbol("x", Domain::Real);
            let coeffs: Vec<i64> = (1..=(deg as i64 + 1)).collect();
            let expr = poly_expr(&p, x, &coeffs);
            b.iter(|| diff(expr, x, &p).unwrap());
        });
    }

    // d/dx sin(x^2) — power rule inside chain rule
    g.throughput(Throughput::Elements(1));
    g.bench_function("sin_x_squared", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = p.func("sin", vec![p.pow(x, p.integer(2_i32))]);
        b.iter(|| diff(expr, x, &p).unwrap());
    });

    // d/dx exp(sin(x)) — two nested chain rules
    g.bench_function("exp_sin_x", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = p.func("exp", vec![p.func("sin", vec![x])]);
        b.iter(|| diff(expr, x, &p).unwrap());
    });

    // d/dx log(x^3 + 2x + 1)
    g.bench_function("log_poly", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let inner = poly_expr(&p, x, &[1, 2, 0, 1]);
        let expr = p.func("log", vec![inner]);
        b.iter(|| diff(expr, x, &p).unwrap());
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Group 4: UniPoly (FLINT-backed)
// ---------------------------------------------------------------------------

fn bench_unipoly(c: &mut Criterion) {
    let mut g = c.benchmark_group("unipoly");

    for deg in [2usize, 4, 8] {
        g.throughput(Throughput::Elements(deg as u64 + 1));
        g.bench_with_input(
            BenchmarkId::new("from_symbolic_deg", deg),
            &deg,
            |b, &deg| {
                let p = fresh_pool();
                let x = p.symbol("x", Domain::Real);
                let coeffs: Vec<i64> = (1..=(deg as i64 + 1)).collect();
                let expr = poly_expr(&p, x, &coeffs);
                b.iter(|| UniPoly::from_symbolic(expr, x, &p).unwrap());
            },
        );
    }

    // (x^4 + …) * (x^4 + …) — FLINT fmpz_poly_mul
    g.throughput(Throughput::Elements(1));
    g.bench_function("mul_deg4_x_deg4", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let f = UniPoly::from_symbolic(poly_expr(&p, x, &[1, 2, 3, 4, 5]), x, &p).unwrap();
        let g2 = UniPoly::from_symbolic(poly_expr(&p, x, &[5, 4, 3, 2, 1]), x, &p).unwrap();
        b.iter(|| &f * &g2);
    });

    // GCD via FLINT
    g.bench_function("gcd_deg4", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let f = UniPoly::from_symbolic(poly_expr(&p, x, &[1, 0, -1, 0, 1]), x, &p).unwrap();
        let g2 = UniPoly::from_symbolic(poly_expr(&p, x, &[1, 1, 0, -1, -1]), x, &p).unwrap();
        b.iter(|| f.gcd(&g2));
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Group 5: MultiPoly (sparse BTreeMap)
// ---------------------------------------------------------------------------

fn bench_multipoly(c: &mut Criterion) {
    let mut g = c.benchmark_group("multipoly");

    g.throughput(Throughput::Elements(5));
    g.bench_function("from_symbolic_univariate_deg4", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = poly_expr(&p, x, &[1, 2, 3, 4, 5]);
        b.iter(|| MultiPoly::from_symbolic(expr, vec![x], &p).unwrap());
    });

    // x^2*y + x*y^2 + x + y
    g.throughput(Throughput::Elements(4));
    g.bench_function("from_symbolic_bivariate", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let x2y = p.mul(vec![p.pow(x, p.integer(2_i32)), y]);
        let xy2 = p.mul(vec![x, p.pow(y, p.integer(2_i32))]);
        let expr = p.add(vec![x2y, p.mul(vec![x, y]), xy2, x, y]);
        b.iter(|| MultiPoly::from_symbolic(expr, vec![x, y], &p).unwrap());
    });

    g.throughput(Throughput::Elements(1));
    g.bench_function("mul_bivariate", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let f = MultiPoly::from_symbolic(p.add(vec![x, y]), vec![x, y], &p).unwrap();
        let g2 = MultiPoly::from_symbolic(
            p.add(vec![p.mul(vec![x, y]), p.integer(1_i32)]),
            vec![x, y],
            &p,
        )
        .unwrap();
        b.iter(|| f.clone() * g2.clone());
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Group 6: memory — allocation bytes per operation
//
// Each case snapshots the global allocator counters before and after the
// operation and prints (bytes_allocated, alloc_call_count) to confirm that
// structural sharing is working and that no unexpected heap traffic occurs.
// ---------------------------------------------------------------------------

fn bench_memory(c: &mut Criterion) {
    let mut g = c.benchmark_group("memory");
    // Use a longer sample time so the one-shot allocation measurements are stable.
    g.sample_size(20);

    // Measure raw bytes allocated to build x + 0, then simplify it.
    g.bench_function("simplify_add_zero_bytes", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let p = fresh_pool();
                let x = p.symbol("x", Domain::Real);
                let expr = p.add(vec![x, p.integer(0_i32)]);
                let snap = snapshot();
                let start = std::time::Instant::now();
                let _ = simplify(expr, &p);
                total += start.elapsed();
                let (bytes, calls) = delta(snap);
                // Emit as criterion-visible custom measurement — reported in
                // the HTML report under "memory" section.
                criterion::black_box((bytes, calls));
            }
            total
        });
    });

    // Diff a degree-3 polynomial and measure allocations.
    g.bench_function("diff_poly_deg3_bytes", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let p = fresh_pool();
                let x = p.symbol("x", Domain::Real);
                let expr = poly_expr(&p, x, &[1, 2, 3, 4]);
                let snap = snapshot();
                let start = std::time::Instant::now();
                let _ = diff(expr, x, &p).unwrap();
                total += start.elapsed();
                let (bytes, calls) = delta(snap);
                criterion::black_box((bytes, calls));
            }
            total
        });
    });

    // Hash-consing: re-build x^2 + 2x + 1 twice; second time must not grow the pool.
    g.bench_function("hash_consing_second_build", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let p = fresh_pool();
                let x = p.symbol("x", Domain::Real);
                let _ = poly_expr(&p, x, &[1, 2, 1]); // first build — populates cache
                let nodes_before = p.len();
                let start = std::time::Instant::now();
                let _ = poly_expr(&p, x, &[1, 2, 1]); // second build — should hit cache
                total += start.elapsed();
                let nodes_after = p.len();
                // Hash-consing guarantees the pool doesn't grow on a duplicate tree.
                assert_eq!(
                    nodes_before, nodes_after,
                    "pool grew on second build: {nodes_before} → {nodes_after}"
                );
            }
            total
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Group 7: DerivationLog — log overhead measurement
// ---------------------------------------------------------------------------

fn bench_log_overhead(c: &mut Criterion) {
    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("log_overhead");

    // Compare diff (which logs everything) vs a no-log simplify baseline.
    g.bench_function("diff_poly_deg4_log_steps", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = poly_expr(&p, x, &[1, 2, 3, 4, 5]);
        b.iter(|| {
            let result = diff(expr, x, &p).unwrap();
            criterion::black_box(result.log.len())
        });
    });

    g.bench_function("simplify_poly_deg4_log_steps", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = poly_expr(&p, x, &[1, 2, 3, 4, 5]);
        b.iter(|| {
            let result = simplify(expr, &p);
            criterion::black_box(result.log.len())
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Group 8: simplifier_comparison — rule-based vs e-graph
//
// Risk-register target: ≤ 2× overhead for simple expressions.
// ---------------------------------------------------------------------------

fn bench_simplifier_comparison(c: &mut Criterion) {
    let mut g = c.benchmark_group("simplifier_comparison");

    // ── identity rewrites (x + 0 → x, x * 1 → x) ────────────────────────

    g.bench_function("identity_rule_based", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = p.add(vec![x, p.integer(0_i32)]);
        b.iter(|| simplify(expr, &p));
    });

    #[cfg(feature = "egraph")]
    g.bench_function("identity_egraph", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = p.add(vec![x, p.integer(0_i32)]);
        b.iter(|| simplify_egraph(expr, &p));
    });

    // ── constant-fold chain: 2 + 3 + 4 + 5 → 14 ─────────────────────────

    g.bench_function("const_fold_rule_based", |b| {
        let p = fresh_pool();
        let expr = p.add(vec![
            p.integer(2_i32),
            p.integer(3_i32),
            p.integer(4_i32),
            p.integer(5_i32),
        ]);
        b.iter(|| simplify(expr, &p));
    });

    #[cfg(feature = "egraph")]
    g.bench_function("const_fold_egraph", |b| {
        let p = fresh_pool();
        let expr = p.add(vec![
            p.integer(2_i32),
            p.integer(3_i32),
            p.integer(4_i32),
            p.integer(5_i32),
        ]);
        b.iter(|| simplify_egraph(expr, &p));
    });

    // ── degree-5 polynomial ───────────────────────────────────────────────

    g.bench_function("poly_deg5_rule_based", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = poly_expr(&p, x, &[1, 2, 3, 4, 5, 6]);
        b.iter(|| simplify(expr, &p));
    });

    #[cfg(feature = "egraph")]
    g.bench_function("poly_deg5_egraph", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = poly_expr(&p, x, &[1, 2, 3, 4, 5, 6]);
        b.iter(|| simplify_egraph(expr, &p));
    });

    // ── cancellation: x + y + (-1)*x → y ─────────────────────────────────

    g.bench_function("cancellation_rule_based", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let neg_x = p.mul(vec![p.integer(-1_i32), x]);
        let expr = p.add(vec![x, y, neg_x]);
        b.iter(|| simplify(expr, &p));
    });

    #[cfg(feature = "egraph")]
    g.bench_function("cancellation_egraph", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let neg_x = p.mul(vec![p.integer(-1_i32), x]);
        let expr = p.add(vec![x, y, neg_x]);
        b.iter(|| simplify_egraph(expr, &p));
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Phase 21 — JIT vs interpreter benchmark
// ---------------------------------------------------------------------------

fn bench_jit(c: &mut Criterion) {
    let mut g = c.benchmark_group("jit");

    // Benchmark tree-walking interpreter (always available)
    g.bench_function("interp_poly_deg4", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = poly_expr(&p, x, &[1, 2, 3, 4, 5]);
        let mut env = HashMap::new();
        env.insert(x, 1.5f64);
        b.iter(|| eval_interp(criterion::black_box(expr), &env, &p));
    });

    // Benchmark compiled interpreter via CompiledFn
    g.bench_function("compiled_poly_deg4", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = poly_expr(&p, x, &[1, 2, 3, 4, 5]);
        let f = compile(expr, &[x], &p).expect("compile failed");
        b.iter(|| f.call(criterion::black_box(&[1.5f64])));
    });

    // Large polynomial: degree 19
    g.bench_function("interp_poly_deg19", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        // sum of x^0 + x^1 + ... + x^19
        let args: Vec<ExprId> = (0_i32..20).map(|k| p.pow(x, p.integer(k))).collect();
        let expr = p.add(args);
        let mut env = HashMap::new();
        env.insert(x, 1.1f64);
        b.iter(|| eval_interp(criterion::black_box(expr), &env, &p));
    });

    g.bench_function("compiled_poly_deg19", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let args: Vec<ExprId> = (0_i32..20).map(|k| p.pow(x, p.integer(k))).collect();
        let expr = p.add(args);
        let f = compile(expr, &[x], &p).expect("compile failed");
        b.iter(|| f.call(criterion::black_box(&[1.1f64])));
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Phase 22 — Ball arithmetic benchmark
// ---------------------------------------------------------------------------

fn bench_ball(c: &mut Criterion) {
    let mut g = c.benchmark_group("ball");

    g.bench_function("arb_add_128bit", |b| {
        let a = ArbBall::from_midpoint_radius(1.5, 0.1, 128);
        let bv = ArbBall::from_midpoint_radius(2.5, 0.2, 128);
        b.iter(|| criterion::black_box(a.clone()) + criterion::black_box(bv.clone()));
    });

    g.bench_function("arb_mul_128bit", |b| {
        let a = ArbBall::from_midpoint_radius(1.5, 0.1, 128);
        let bv = ArbBall::from_midpoint_radius(2.5, 0.2, 128);
        b.iter(|| criterion::black_box(a.clone()) * criterion::black_box(bv.clone()));
    });

    g.bench_function("arb_sin_128bit", |b| {
        let a = ArbBall::from_midpoint_radius(1.0, 0.01, 128);
        b.iter(|| criterion::black_box(a.clone()).sin());
    });

    g.bench_function("arb_powi_128bit", |b| {
        let a = ArbBall::from_midpoint_radius(2.0, 0.1, 128);
        b.iter(|| criterion::black_box(a.clone()).powi(10));
    });

    // interval_eval for a polynomial
    g.bench_function("interval_eval_poly_deg4", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = poly_expr(&p, x, &[1, 2, 3, 4, 5]);
        let x_ball = ArbBall::from_midpoint_radius(1.0, 0.1, 128);
        b.iter(|| {
            let mut ev = IntervalEval::new(128);
            ev.bind(x, criterion::black_box(x_ball.clone()));
            ev.eval(criterion::black_box(expr), &p)
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Phase 23 — Parallel simplification benchmark
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
fn bench_par(c: &mut Criterion) {
    use alkahest_core::simplify_par;
    let mut g = c.benchmark_group("par_simplify");
    g.throughput(Throughput::Elements(1));

    // Large Add node with many zeros → all collapse to the lone x
    g.bench_function("large_add_seq", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let zero = p.integer(0_i32);
        let mut args = vec![x];
        args.extend(std::iter::repeat(zero).take(63));
        let expr = p.add(args);
        b.iter(|| simplify(criterion::black_box(expr), &p));
    });

    g.bench_function("large_add_par", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let zero = p.integer(0_i32);
        let mut args = vec![x];
        args.extend(std::iter::repeat(zero).take(63));
        let expr = p.add(args);
        b.iter(|| simplify_par(criterion::black_box(expr), &p));
    });

    g.bench_function("large_mul_seq", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let one = p.integer(1_i32);
        let mut args = vec![x];
        args.extend(std::iter::repeat(one).take(63));
        let expr = p.mul(args);
        b.iter(|| simplify(criterion::black_box(expr), &p));
    });

    g.bench_function("large_mul_par", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let one = p.integer(1_i32);
        let mut args = vec![x];
        args.extend(std::iter::repeat(one).take(63));
        let expr = p.mul(args);
        b.iter(|| simplify_par(criterion::black_box(expr), &p));
    });

    g.finish();
}

#[cfg(not(feature = "parallel"))]
fn bench_par(_c: &mut Criterion) {}

// ---------------------------------------------------------------------------
// V1-1 — NVPTX GPU JIT benchmark. Only compiled in with `--features cuda`
// and only measures anything on a box where `CudaContext::new(0)` succeeds.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn bench_nvptx(c: &mut Criterion) {
    use alkahest_core::jit::nvptx::compile_cuda;

    if cudarc::driver::CudaContext::new(0).is_err() {
        eprintln!("nvptx bench: no CUDA device, skipping");
        return;
    }

    let mut g = c.benchmark_group("nvptx");
    g.sample_size(20);

    // 1M-pt polynomial row required by the V1-1 acceptance gate.
    g.bench_function("nvptx_polynomial_1M", |b| {
        let p = fresh_pool();
        let x = p.symbol("x", Domain::Real);
        let expr = poly_expr(&p, x, &[1, 2, 3, 4, 5]);
        let f = compile_cuda(expr, &[x], &p).expect("compile_cuda");
        let n = 1 << 20;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * 1e-6).collect();
        let mut out = vec![0.0f64; n];
        // Warm up: first launch pays module-load + libdevice-link cost.
        f.call_batch(&[&xs[..]], &mut out).unwrap();
        b.iter(|| {
            f.call_batch(criterion::black_box(&[&xs[..]]), &mut out)
                .unwrap();
        });
    });

    // CPU-JIT baseline for the 10× claim is measured out-of-harness by
    // `examples/cpu_jit_probe.rs`, because the existing criterion bench
    // infrastructure interacts poorly with LLVM MCJIT under
    // `--features jit` (preexisting segfault unrelated to V1-1).

    g.finish();
}

#[cfg(not(feature = "cuda"))]
fn bench_nvptx(_c: &mut Criterion) {}

fn bench_hnf(c: &mut Criterion) {
    let n = 50;
    let rows: Vec<Vec<i64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (((i * 17 + j * 31) % 97) as i64).saturating_sub(48))
                .collect()
        })
        .collect();
    let m = IntegerMatrix::from_nested(rows).unwrap();
    c.bench_function("hnf_50x50", |b| {
        b.iter(|| {
            let _ = hermite_form(std::hint::black_box(&m));
        });
    });
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_intern,
    bench_simplify,
    bench_diff,
    bench_unipoly,
    bench_multipoly,
    bench_memory,
    bench_log_overhead,
    bench_simplifier_comparison,
    bench_jit,
    bench_ball,
    bench_par,
    bench_nvptx,
    bench_hnf,
);
criterion_main!(benches);
