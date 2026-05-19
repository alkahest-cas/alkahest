use alkahest_cas::jit::compile;
use alkahest_cas::kernel::{Domain, ExprPool};
use std::time::Instant;

fn main() {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Real);
    let mut terms = vec![p.integer(1)];
    for (i, &c) in [2i64, 3, 4, 5].iter().enumerate() {
        let xpow = p.pow(x, p.integer((i + 1) as i32));
        terms.push(p.mul(vec![p.integer(c), xpow]));
    }
    let expr = p.add(terms);
    let f = compile(expr, &[x], &p).expect("compile");

    let n = 1 << 20;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 * 1e-6).collect();
    let mut out = vec![0.0f64; n];

    // Warm up.
    f.call_batch(&[&xs[..]], &mut out);

    let t0 = Instant::now();
    let iters = 4;
    for _ in 0..iters {
        f.call_batch(&[&xs[..]], &mut out);
    }
    let per = t0.elapsed() / iters;
    eprintln!(
        "cpu_jit call_batch poly_1M: {:.2} ms/call",
        per.as_secs_f64() * 1e3
    );
}
