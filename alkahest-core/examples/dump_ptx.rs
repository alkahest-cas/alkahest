use alkahest_cas::jit::nvptx::compile_cuda;
use alkahest_cas::kernel::{Domain, ExprPool};
fn main() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let expr = pool.add(vec![pool.mul(vec![x, x]), pool.integer(1)]);
    let c = compile_cuda(expr, &[x], &pool).unwrap();
    let bytes = c.ptx.as_bytes();
    std::fs::write("/tmp/dump.ptx", bytes).unwrap();
    eprintln!(
        "len={} nul_count={}",
        bytes.len(),
        bytes.iter().filter(|&&b| b == 0).count()
    );
    for (i, &b) in bytes.iter().enumerate() {
        if b == 0 {
            eprintln!("nul at {i}");
        }
    }
}
