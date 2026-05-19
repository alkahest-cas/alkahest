//! Alkahest-cas quickstart — demonstrates the core Rust API.
//!
//! Run from the repo root:
//!   cargo run --manifest-path examples/rust_quickstart/Cargo.toml
//!
//! Once the crate is on crates.io, swap the path dep for:
//!   alkahest-cas = "2"

use alkahest_cas::kernel::Domain;
use alkahest_cas::number_theory;
use alkahest_cas::poly::UniPoly;
use alkahest_cas::{diff, integrate, parse, render_latex, render_unicode, simplify, ExprPool};
use std::collections::HashMap;

fn main() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);

    // -----------------------------------------------------------------------
    // 1. Parse an expression from a string
    // -----------------------------------------------------------------------
    let mut syms = HashMap::from([("x".to_owned(), x)]);
    let expr = parse("sin(x)^2 + cos(x)^2", &pool, &mut syms).unwrap();
    println!("parsed:    {}", render_unicode(expr, &pool));

    // -----------------------------------------------------------------------
    // 2. Simplification with derivation log
    // -----------------------------------------------------------------------
    let zero = pool.integer(0i64);
    let r = simplify(pool.add(vec![x, zero]), &pool);
    println!(
        "simplify:  {}  ({} step(s))",
        render_unicode(r.value, &pool),
        r.log.len()
    );

    // -----------------------------------------------------------------------
    // 3. Symbolic differentiation
    // -----------------------------------------------------------------------
    let f = pool.func("sin", vec![pool.pow(x, pool.integer(2i64))]); // sin(x^2)
    let d = diff(f, x, &pool).unwrap();
    println!("diff:      {}", render_latex(d.value, &pool));

    // -----------------------------------------------------------------------
    // 4. Symbolic integration
    // -----------------------------------------------------------------------
    let i = integrate(pool.func("exp", vec![x]), x, &pool).unwrap();
    println!("integrate: {}", render_unicode(i.value, &pool));

    // -----------------------------------------------------------------------
    // 5. Univariate polynomial GCD (FLINT-backed)
    // -----------------------------------------------------------------------
    // x^2 - 1 = (x-1)(x+1); gcd with (x-1) should give (x-1)
    let m1 = pool.integer(-1i64);
    let p_expr = pool.add(vec![pool.pow(x, pool.integer(2i64)), m1]); // x^2 - 1
    let q_expr = pool.add(vec![x, m1]); // x - 1
    let p = UniPoly::from_symbolic(p_expr, x, &pool).unwrap();
    let q = UniPoly::from_symbolic(q_expr, x, &pool).unwrap();
    if let Some(g) = p.gcd(&q) {
        println!("gcd:       {}", g);
    }

    // -----------------------------------------------------------------------
    // 6. Number theory
    // -----------------------------------------------------------------------
    let mersenne = "170141183460469231731687303715884105727"; // 2^127 - 1
    println!(
        "isprime(2^127-1): {}",
        number_theory::isprime(mersenne).unwrap()
    );
}
