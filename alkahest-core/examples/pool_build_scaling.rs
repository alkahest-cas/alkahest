//! Does `ExprPool::mul` scale linearly in the size of the expression?
//!
//! `pool.mul` calls `mult_tree_is_commutative`, which walks the *entire*
//! subtree on every call.  `pool.add` does no such walk.  If that walk is the
//! cost, building a nested product should grow quadratically while a nested
//! sum grows linearly.

use alkahest_cas::kernel::{Domain, ExprId, ExprPool};
use std::time::Instant;

fn chain_mul(pool: &ExprPool, depth: usize) -> ExprId {
    let one = pool.integer(1);
    let mut e = pool.symbol("x", Domain::Real);
    for _ in 0..depth {
        e = pool.mul(vec![e, one]);
    }
    e
}

fn chain_add(pool: &ExprPool, depth: usize) -> ExprId {
    let zero = pool.integer(0);
    let mut e = pool.symbol("x", Domain::Real);
    for _ in 0..depth {
        e = pool.add(vec![e, zero]);
    }
    e
}

fn main() {
    println!(
        "{:>8} {:>12} {:>12}   {:>10}",
        "depth", "mul (ms)", "add (ms)", "mul/add"
    );
    let mut prev_mul = 0.0_f64;
    for depth in [500usize, 1000, 2000, 4000, 8000] {
        let pool = ExprPool::new();
        let t = Instant::now();
        std::hint::black_box(chain_mul(&pool, depth));
        let mul = t.elapsed().as_secs_f64() * 1e3;

        let pool2 = ExprPool::new();
        let t = Instant::now();
        std::hint::black_box(chain_add(&pool2, depth));
        let add = t.elapsed().as_secs_f64() * 1e3;

        let growth = if prev_mul > 0.0 {
            format!("{:.1}x vs prev depth", mul / prev_mul)
        } else {
            String::new()
        };
        prev_mul = mul;
        println!(
            "{depth:>8} {mul:>12.2} {add:>12.2}   {:>10.1}x  {growth}",
            mul / add
        );
    }
}
