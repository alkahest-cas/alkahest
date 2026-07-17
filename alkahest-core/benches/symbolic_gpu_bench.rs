//! Reproducible CPU baseline + optional GPU comparison for candidate symbolic GPU kernels.
//!
//! Records structured metrics (input family, size, coeff bits, wall time, heap delta,
//! correctness) and applies crossover-policy scaffolding. GPU is **not** treated as
//! default success — see `docs/symbolic-gpu-benchmarks.md`.
//!
//! # Run
//!
//! ```bash
//! # CPU baseline only (default features; runs on CI)
//! cargo bench -p alkahest-cas --bench symbolic_gpu_bench -- --nocapture
//!
//! # Include Macaulay-matrix mod-p row reduction + optional GPU path
//! cargo bench -p alkahest-cas --bench symbolic_gpu_bench \
//!   --features groebner-cuda -- --nocapture
//!
//! # With an NVIDIA GPU present, set ALKAHEST_GPU_BENCH=1 to time the CUDA path
//! ALKAHEST_GPU_BENCH=1 cargo bench -p alkahest-cas --bench symbolic_gpu_bench \
//!   --features groebner-cuda -- --nocapture
//! ```

use alkahest_cas::{Domain, ExprPool, UniPoly};
use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Metrics schema
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct BenchRecord {
    kernel_family: &'static str,
    backend: &'static str,
    input_family: &'static str,
    size: usize,
    coeff_bits: u32,
    wall_ns: u64,
    alloc_bytes: u64,
    correct: bool,
    note: Option<&'static str>,
}

impl BenchRecord {
    fn to_json_line(&self) -> String {
        let note = self
            .note
            .map(|n| format!(",\"note\":\"{n}\""))
            .unwrap_or_default();
        format!(
            "{{\"kernel_family\":\"{}\",\"backend\":\"{}\",\"input_family\":\"{}\",\
             \"size\":{},\"coeff_bits\":{},\"wall_ns\":{},\"alloc_bytes\":{},\
             \"correct\":{}{}}}",
            self.kernel_family,
            self.backend,
            self.input_family,
            self.size,
            self.coeff_bits,
            self.wall_ns,
            self.alloc_bytes,
            self.correct,
            note
        )
    }
}

// ---------------------------------------------------------------------------
// Crossover policy scaffolding (not wired into production dispatch)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct CrossoverPolicy {
    /// Require GPU to beat CPU by at least this factor before recommending GPU.
    min_gpu_speedup: f64,
    /// Ignore GPU timing below this problem size.
    min_size: usize,
}

impl Default for CrossoverPolicy {
    fn default() -> Self {
        Self {
            min_gpu_speedup: 1.10,
            min_size: 1,
        }
    }
}

fn crossover_recommendation(
    records: &[BenchRecord],
    policy: &CrossoverPolicy,
) -> Vec<(String, String, usize)> {
    let mut by_key: std::collections::BTreeMap<(String, usize), (Option<u64>, Option<u64>)> =
        std::collections::BTreeMap::new();

    for r in records {
        if !r.correct {
            continue;
        }
        let key = (r.kernel_family.to_string(), r.size);
        let entry = by_key.entry(key).or_insert((None, None));
        match r.backend {
            "cpu" => entry.0 = Some(r.wall_ns),
            "gpu" => entry.1 = Some(r.wall_ns),
            _ => {}
        }
    }

    let mut out = Vec::new();
    for ((family, size), (cpu_ns, gpu_ns)) in by_key {
        let (Some(cpu), Some(gpu)) = (cpu_ns, gpu_ns) else {
            continue;
        };
        if size < policy.min_size {
            out.push((family, "cpu".into(), size));
            continue;
        }
        let speedup = cpu as f64 / gpu as f64;
        let pick = if speedup >= policy.min_gpu_speedup {
            "gpu"
        } else {
            "cpu"
        };
        out.push((family, pick.into(), size));
    }
    out
}

// ---------------------------------------------------------------------------
// Cheap heap counter (same idea as alkahest_bench.rs)
// ---------------------------------------------------------------------------

static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            ALLOC_BYTES.fetch_add((new_size - layout.size()) as u64, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn alloc_snapshot() -> u64 {
    ALLOC_BYTES.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

const WARMUP_ITERS: u32 = 3;
const MEASURE_ITERS: u32 = 20;

fn time_iters<F: FnMut()>(mut f: F) -> (Duration, u64) {
    for _ in 0..WARMUP_ITERS {
        f();
    }
    let before = alloc_snapshot();
    let start = Instant::now();
    for _ in 0..MEASURE_ITERS {
        f();
    }
    let elapsed = start.elapsed();
    let alloc_delta = alloc_snapshot().saturating_sub(before);
    (elapsed / MEASURE_ITERS, alloc_delta)
}

#[cfg(feature = "groebner-cuda")]
fn gpu_bench_enabled() -> bool {
    env::var("ALKAHEST_GPU_BENCH").ok().as_deref() == Some("1")
}

// ---------------------------------------------------------------------------
// Kernel 1 — UniPoly multiply (FLINT fmpz_poly_mul; NTT/FFT internally)
// ---------------------------------------------------------------------------

fn poly_expr(p: &ExprPool, x: alkahest_cas::ExprId, coeffs: &[i64]) -> alkahest_cas::ExprId {
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

fn random_coeffs(deg: usize, seed: u64) -> Vec<i64> {
    let mut x = seed;
    (0..=deg)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((x >> 33) as i64 % 97).saturating_sub(48)
        })
        .collect()
}

fn coeff_bit_width(coeffs: &[i64]) -> u32 {
    coeffs
        .iter()
        .map(|&c| {
            let v = c.unsigned_abs();
            if v == 0 {
                0
            } else {
                64 - v.leading_zeros()
            }
        })
        .max()
        .unwrap_or(0)
}

fn bench_unipoly_mul() -> Vec<BenchRecord> {
    let degrees = [256usize, 1024, 4096];
    let mut out = Vec::new();

    for &deg in &degrees {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let a_coeffs = random_coeffs(deg, deg as u64 + 1);
        let b_coeffs = random_coeffs(deg, deg as u64 + 17);
        let coeff_bits = coeff_bit_width(&a_coeffs).max(coeff_bit_width(&b_coeffs));

        let a = UniPoly::from_symbolic(poly_expr(&p, x, &a_coeffs), x, &p).unwrap();
        let b = UniPoly::from_symbolic(poly_expr(&p, x, &b_coeffs), x, &p).unwrap();

        // Reference multiply for correctness (once).
        let reference = &a * &b;

        let (wall, alloc_bytes) = time_iters(|| {
            let prod = &a * &b;
            std::hint::black_box(prod.degree());
        });

        let check = &a * &b;
        let correct = check == reference;

        out.push(BenchRecord {
            kernel_family: "unipoly_mul",
            backend: "cpu",
            input_family: "random_integer_coeffs",
            size: deg,
            coeff_bits,
            wall_ns: wall.as_nanos() as u64,
            alloc_bytes,
            correct,
            note: Some("FLINT fmpz_poly_mul (NTT/FFT for large degrees)"),
        });
    }

    out
}

// ---------------------------------------------------------------------------
// Kernel 2 — Macaulay matrix row reduction mod p (groebner-cuda feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "groebner-cuda")]
fn bench_macaulay_reduce() -> Vec<BenchRecord> {
    use alkahest_cas::poly::groebner::cuda::MacaulayMatrix;
    use alkahest_cas::poly::groebner::ideal::GbPoly;
    use alkahest_cas::poly::groebner::MonomialOrder;
    use rug::Rational;

    const PRIME: u64 = 2_147_483_647;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }

    fn poly(terms: &[(&[u32], i64)]) -> GbPoly {
        let n_vars = terms.first().map(|(e, _)| e.len()).unwrap_or(1);
        GbPoly {
            terms: terms.iter().map(|(e, c)| (e.to_vec(), rat(*c))).collect(),
            n_vars,
        }
    }

    // Toy systems: row count scales with `n` (Katsura-style dense coupling).
    let systems: &[(&str, Vec<GbPoly>)] = &[
        (
            "circle_line",
            vec![
                poly(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -1)]),
                poly(&[(&[0, 1], 1), (&[1, 0], -1)]),
            ],
        ),
        (
            "katsura2",
            vec![
                poly(&[
                    (&[1, 0, 0], 1),
                    (&[0, 1, 0], 1),
                    (&[0, 0, 1], 1),
                    (&[0, 0, 0], -1),
                ]),
                poly(&[
                    (&[1, 0, 0], 1),
                    (&[0, 1, 0], 1),
                    (&[0, 0, 1], -1),
                    (&[0, 0, 0], 1),
                ]),
                poly(&[
                    (&[2, 0, 0], 1),
                    (&[0, 2, 0], 1),
                    (&[0, 0, 2], 1),
                    (&[0, 0, 0], -1),
                ]),
            ],
        ),
    ];

    let mut out = Vec::new();

    for (input_family, polys) in systems {
        let size = polys.len();
        let coeff_bits = 32; // 31-bit CRT primes; coefficients are small integers

        let reference = {
            let mut m = MacaulayMatrix::build(polys, MonomialOrder::Lex, PRIME)
                .expect("unlucky prime for toy system");
            m.reduce_cpu();
            m.data
        };

        let (cpu_wall, cpu_alloc) = time_iters(|| {
            let mut m = MacaulayMatrix::build(polys, MonomialOrder::Lex, PRIME)
                .expect("unlucky prime for toy system");
            m.reduce_cpu();
            std::hint::black_box(&m.data[..]);
        });

        out.push(BenchRecord {
            kernel_family: "macaulay_reduce_mod_p",
            backend: "cpu",
            input_family,
            size,
            coeff_bits,
            wall_ns: cpu_wall.as_nanos() as u64,
            alloc_bytes: cpu_alloc,
            correct: true,
            note: Some("dense GF(p) RREF; Gröbner Macaulay preprocessing"),
        });

        if gpu_bench_enabled() {
            if cudarc::driver::CudaContext::new(0).is_err() {
                out.push(BenchRecord {
                    kernel_family: "macaulay_reduce_mod_p",
                    backend: "gpu",
                    input_family,
                    size,
                    coeff_bits,
                    wall_ns: 0,
                    alloc_bytes: 0,
                    correct: false,
                    note: Some("skipped: no CUDA device"),
                });
            } else {
                let (gpu_wall, gpu_alloc) = time_iters(|| {
                    let mut m = MacaulayMatrix::build(polys, MonomialOrder::Lex, PRIME)
                        .expect("unlucky prime for toy system");
                    m.reduce_gpu(0).expect("GPU reduce");
                    std::hint::black_box(&m.data[..]);
                });

                let mut check = MacaulayMatrix::build(polys, MonomialOrder::Lex, PRIME)
                    .expect("unlucky prime for toy system");
                check.reduce_gpu(0).expect("GPU reduce check");
                let correct = check.data == reference;

                out.push(BenchRecord {
                    kernel_family: "macaulay_reduce_mod_p",
                    backend: "gpu",
                    input_family,
                    size,
                    coeff_bits,
                    wall_ns: gpu_wall.as_nanos() as u64,
                    alloc_bytes: gpu_alloc,
                    correct,
                    note: Some("PTX eliminate_row_kernel; host pivot loop"),
                });
            }
        }
    }

    out
}

#[cfg(not(feature = "groebner-cuda"))]
fn bench_macaulay_reduce() -> Vec<BenchRecord> {
    Vec::new()
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

fn report_path() -> PathBuf {
    env::var("SYMBOLIC_GPU_BENCH_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/symbolic_gpu_bench.jsonl"))
}

fn write_report(records: &[BenchRecord]) {
    if let Some(parent) = report_path().parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut body = String::new();
    for r in records {
        body.push_str(&r.to_json_line());
        body.push('\n');
    }
    fs::write(report_path(), &body).expect("write benchmark report");
    eprintln!("wrote {}", report_path().display());
    for r in records {
        eprintln!(
            "  {:24} {:4} {:20} size={:4} coeff_bits={:2} {:8.3} ms alloc={:8} correct={}",
            r.kernel_family,
            r.backend,
            r.input_family,
            r.size,
            r.coeff_bits,
            r.wall_ns as f64 / 1e6,
            r.alloc_bytes,
            r.correct,
        );
    }
}

fn main() {
    let mut records = Vec::new();
    records.extend(bench_unipoly_mul());
    records.extend(bench_macaulay_reduce());

    let all_correct = records
        .iter()
        .all(|r| r.correct || r.note == Some("skipped: no CUDA device"));
    if !all_correct {
        eprintln!("correctness check FAILED for at least one record");
        for r in &records {
            if !r.correct && r.note != Some("skipped: no CUDA device") {
                eprintln!(
                    "  FAIL: {:?} {:?} size={}",
                    r.kernel_family, r.backend, r.size
                );
            }
        }
        std::process::exit(1);
    }

    write_report(&records);

    let policy = CrossoverPolicy::default();
    let picks = crossover_recommendation(&records, &policy);
    if picks.is_empty() {
        eprintln!(
            "crossover policy: no CPU/GPU pairs recorded (set ALKAHEST_GPU_BENCH=1 with --features groebner-cuda to compare)"
        );
    } else {
        eprintln!(
            "crossover policy (min_speedup={:.2}, min_size={}):",
            policy.min_gpu_speedup, policy.min_size
        );
        for (family, backend, size) in picks {
            eprintln!("  {family} size={size} -> use {backend}");
        }
    }
}
