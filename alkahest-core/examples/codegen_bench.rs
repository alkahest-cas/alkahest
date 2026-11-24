//! Codegen timing for the compile optimiser: `poly20` (expanded
//! `(1+x+y)^20`) and `trig30` (`Σ sin(kx)cos(ky)/k + exp(-(x²+y²))·log(1+xy)`)
//! compiled exactly as written, with the Horner pass, and with Horner plus
//! common-pair elimination; then compile times of large random polynomials.
//!
//! ```text
//! RAYON_NUM_THREADS=1 cargo run --release --example codegen_bench --features cranelift [N_POINTS]
//! ```

use alkahest_cas::jit::{compile_with_options, CodegenOptions, CompileConfig, CompileTier};
use alkahest_cas::kernel::{Domain, ExprId, ExprPool};
use std::time::Instant;

fn dense(pool: &ExprPool, x: ExprId, y: ExprId, n: u32) -> ExprId {
    let fact = |k: u32| (1..=k).fold(rug::Integer::from(1), |a, b| a * b);
    let mut terms = Vec::new();
    for i in 0..=n {
        for j in 0..=(n - i) {
            let c = fact(n) / (fact(i) * fact(j) * fact(n - i - j));
            let mut f = vec![pool.integer(c)];
            if i > 0 {
                f.push(pool.pow(x, pool.integer(i)));
            }
            if j > 0 {
                f.push(pool.pow(y, pool.integer(j)));
            }
            terms.push(pool.mul(f));
        }
    }
    pool.add(terms)
}

fn trig30(pool: &ExprPool, x: ExprId, y: ExprId) -> ExprId {
    let mut terms: Vec<ExprId> = (1..=30)
        .map(|k| {
            let kx = pool.mul(vec![pool.integer(k), x]);
            let ky = pool.mul(vec![pool.integer(k), y]);
            pool.mul(vec![
                pool.func("sin", vec![kx]),
                pool.func("cos", vec![ky]),
                pool.rational(1, k),
            ])
        })
        .collect();
    let r2 = pool.add(vec![
        pool.pow(x, pool.integer(2)),
        pool.pow(y, pool.integer(2)),
    ]);
    let g = pool.func("exp", vec![pool.mul(vec![pool.integer(-1), r2])]);
    let l = pool.func(
        "log",
        vec![pool.add(vec![pool.integer(1), pool.mul(vec![x, y])])],
    );
    terms.push(pool.mul(vec![g, l]));
    pool.add(terms)
}

/// Random dense-ish polynomial in three variables with `n` terms.
fn random_poly(pool: &ExprPool, vars: &[ExprId], n: usize, seed: u64) -> ExprId {
    let mut s = seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut seen = std::collections::HashSet::new();
    let mut terms = Vec::new();
    let deg = ((n as f64).cbrt() * 1.9) as u64 + 2;
    while terms.len() < n {
        let e: Vec<u64> = (0..3).map(|_| next() % deg).collect();
        if !seen.insert(e.clone()) {
            continue;
        }
        let mut f = vec![pool.integer((next() % 2001) as i64 - 1000)];
        for (k, &ek) in e.iter().enumerate() {
            if ek > 0 {
                f.push(pool.pow(vars[k], pool.integer(ek as i64)));
            }
        }
        terms.push(pool.mul(f));
    }
    pool.add(terms)
}

fn opts(name: &str) -> CodegenOptions {
    match name {
        "exact" => CodegenOptions::exact(),
        "horner" => CodegenOptions::exact().with_horner(true),
        _ => CodegenOptions::exact().with_horner(true).with_cpe(true),
    }
}

fn config() -> CompileConfig {
    CompileConfig {
        expected_evals: Some(1 << 20),
        force_tier: Some(CompileTier::Cranelift),
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1_000_000);
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let z = pool.symbol("z", Domain::Real);
    let mut state = 7u64;
    let mut unit = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let xs: Vec<f64> = (0..n).map(|_| 0.05 + 0.9 * unit()).collect();
    let ys: Vec<f64> = (0..n).map(|_| 0.05 + 0.9 * unit()).collect();
    let mut out = vec![0.0; n];

    println!("expr     passes        compile_ms   ns/pt   max_rel_err");
    for (name, e) in [
        ("poly20", dense(&pool, x, y, 20)),
        ("trig30", trig30(&pool, x, y)),
    ] {
        let reference: Vec<f64> = (0..n)
            .map(|i| {
                let (a, b) = (xs[i], ys[i]);
                if name == "poly20" {
                    (1.0 + a + b).powi(20)
                } else {
                    (1..=30)
                        .map(|k| {
                            let k = f64::from(k);
                            (k * a).sin() * (k * b).cos() / k
                        })
                        .sum::<f64>()
                        + (-(a * a + b * b)).exp() * (a * b).ln_1p()
                }
            })
            .collect();
        for passes in ["exact", "horner", "horner+cpe"] {
            let mut tc = f64::INFINITY;
            let mut f = None;
            for _ in 0..7 {
                let t0 = Instant::now();
                let g = compile_with_options(e, &[x, y], &pool, config(), opts(passes)).unwrap();
                tc = tc.min(t0.elapsed().as_secs_f64());
                f = Some(g);
            }
            let f = f.unwrap();
            let mut te = f64::INFINITY;
            for _ in 0..5 {
                let t0 = Instant::now();
                f.call_batch(&[&xs, &ys], &mut out);
                te = te.min(t0.elapsed().as_secs_f64());
            }
            let err = out
                .iter()
                .zip(&reference)
                .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
                .fold(0.0f64, f64::max);
            println!(
                "{name:8} {passes:12} {:10.3} {:8.2}   {err:.2e}",
                tc * 1e3,
                te / n as f64 * 1e9
            );
        }
    }

    if std::env::args().nth(2).as_deref() == Some("quick") {
        return;
    }

    // Wider polynomials: more independent chains per point.
    println!("\nlarger polynomials, horner+cpe: ns/pt");
    let zs: Vec<f64> = (0..n).map(|_| 0.05 + 0.9 * unit()).collect();
    let w = pool.symbol("w", Domain::Real);
    let one_plus = |vs: &[ExprId]| {
        let mut a = vec![pool.integer(1)];
        a.extend_from_slice(vs);
        pool.add(a)
    };
    let cases = [
        ("(1+x+y+z)^15 expanded", vec![x, y, z], 15u32),
        ("(1+x+y+z+w)^8 expanded", vec![x, y, z, w], 8),
        ("(1+x)^60 expanded", vec![x], 60),
    ];
    for (name, vars, deg) in cases {
        let e = alkahest_cas::poly::MultiPoly::from_symbolic(
            pool.pow(one_plus(&vars), pool.integer(deg)),
            vars.clone(),
            &pool,
        )
        .unwrap()
        .to_expr(&pool);
        let cols: Vec<&[f64]> = [&xs[..], &ys[..], &zs[..], &xs[..]][..vars.len()].to_vec();
        let f = compile_with_options(e, &vars, &pool, config(), opts("horner+cpe")).unwrap();
        let mut te = f64::INFINITY;
        for _ in 0..5 {
            let t0 = Instant::now();
            f.call_batch(&cols, &mut out);
            te = te.min(t0.elapsed().as_secs_f64());
        }
        println!("{name:26} {:8.2}", te / n as f64 * 1e9);
    }
    if std::env::args().nth(2).as_deref() == Some("wide") {
        return;
    }
    println!("\nrandom 3-variable polynomials: compile seconds");
    println!("terms    horner     horner+cpe");
    for terms in [1_000usize, 10_000, 100_000] {
        let e = random_poly(&pool, &[x, y, z], terms, 0x9E37_79B9_7F4A_7C15);
        let mut row = format!("{terms:<8}");
        for passes in ["horner", "horner+cpe"] {
            let t0 = Instant::now();
            let _f = compile_with_options(e, &[x, y, z], &pool, config(), opts(passes)).unwrap();
            row += &format!(" {:9.3}", t0.elapsed().as_secs_f64());
        }
        println!("{row}");
    }
}
