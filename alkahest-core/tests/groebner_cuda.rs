//! Integration tests for the V1-7 GPU-backed Gröbner basis (groebner-cuda feature).
//!
//! The CPU-fallback path (`device_id = None`) exercises the full CRT+rational-
//! reconstruction pipeline without requiring a CUDA device.  The GPU path
//! tests are gated by an environment variable (`ALKAHEST_GPU_TESTS=1`) so CI
//! passes on machines without an NVIDIA GPU.

#![cfg(feature = "groebner-cuda")]

use alkahest_cas::poly::groebner::cuda::{compute_groebner_basis_gpu, MacaulayMatrix};
use alkahest_cas::poly::groebner::f4::compute_groebner_basis;
use alkahest_cas::poly::groebner::ideal::GbPoly;
use alkahest_cas::poly::groebner::monomial_order::MonomialOrder;
use alkahest_cas::poly::groebner::reduce::reduce as cpu_reduce;
use rug::Rational;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn gpu_available() -> bool {
    std::env::var("ALKAHEST_GPU_TESTS").ok().as_deref() == Some("1")
}

// ---------------------------------------------------------------------------
// Correctness — CPU fallback path (no GPU required)
// ---------------------------------------------------------------------------

#[test]
fn linear_system_two_vars() {
    // x + y - 1, x - y  →  solutions x=1/2, y=1/2
    let f = poly(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], -1)]);
    let g = poly(&[(&[1, 0], 1), (&[0, 1], -1)]);
    let basis = compute_groebner_basis_gpu(vec![f.clone(), g.clone()], MonomialOrder::Lex, None)
        .expect("groebner failed");
    assert!(!basis.is_empty(), "basis should not be empty");
    assert!(cpu_reduce(&f, &basis, MonomialOrder::Lex).is_zero());
    assert!(cpu_reduce(&g, &basis, MonomialOrder::Lex).is_zero());
}

#[test]
fn x_squared_minus_1() {
    // (x^2 - 1, x - 1) should give a size-1 basis {x - 1}
    let f = poly(&[(&[2], 1), (&[0], -1)]);
    let g = poly(&[(&[1], 1), (&[0], -1)]);
    let basis = compute_groebner_basis_gpu(vec![f.clone(), g.clone()], MonomialOrder::Lex, None)
        .expect("groebner failed");
    assert_eq!(basis.len(), 1);
    assert!(cpu_reduce(&f, &basis, MonomialOrder::Lex).is_zero());
    assert!(cpu_reduce(&g, &basis, MonomialOrder::Lex).is_zero());
}

#[test]
fn circle_line_intersection() {
    // x^2 + y^2 - 1 = 0, y - x = 0  →  two solutions (±√2/2, ±√2/2)
    let f = poly(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -1)]);
    let g = poly(&[(&[0, 1], 1), (&[1, 0], -1)]);
    let basis = compute_groebner_basis_gpu(vec![f.clone(), g.clone()], MonomialOrder::Lex, None)
        .expect("groebner failed");
    assert!(!basis.is_empty());
    assert!(
        cpu_reduce(&f, &basis, MonomialOrder::Lex).is_zero(),
        "circle not in ideal"
    );
    assert!(
        cpu_reduce(&g, &basis, MonomialOrder::Lex).is_zero(),
        "line not in ideal"
    );
}

#[test]
fn parabola_line() {
    // y - x^2 = 0, y - x = 0  →  x^2 - x = 0, so x=0 or x=1
    let f = poly(&[(&[0, 1], 1), (&[2, 0], -1)]);
    let g = poly(&[(&[0, 1], 1), (&[1, 0], -1)]);
    let basis = compute_groebner_basis_gpu(vec![f.clone(), g.clone()], MonomialOrder::Lex, None)
        .expect("groebner failed");
    assert!(!basis.is_empty());
    assert!(cpu_reduce(&f, &basis, MonomialOrder::Lex).is_zero());
    assert!(cpu_reduce(&g, &basis, MonomialOrder::Lex).is_zero());
}

#[test]
fn inconsistent_system_gives_unit_basis() {
    // (x, x - 1) — inconsistent; Gröbner basis should contain a non-zero constant
    let f = poly(&[(&[1, 0], 1)]);
    let g = poly(&[(&[1, 0], 1), (&[0, 0], -1)]);
    let basis =
        compute_groebner_basis_gpu(vec![f, g], MonomialOrder::Lex, None).expect("groebner failed");
    // The basis for the unit ideal contains 1 (or an element with no free variables)
    let has_constant = basis.iter().any(|b| {
        b.terms.len() == 1
            && b.terms
                .keys()
                .next()
                .map(|e| e.iter().all(|&d| d == 0))
                .unwrap_or(false)
    });
    assert!(has_constant, "inconsistent ideal must yield unit basis");
}

#[test]
fn agrees_with_pure_rust_f4_lex() {
    // Compare GPU (CPU-fallback) vs pure-Rust F4 on a 2-variable quadratic system
    let f = poly(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -1)]); // x^2+y^2-1
    let g = poly(&[(&[0, 1], 1), (&[1, 0], -1)]); // y - x
    let order = MonomialOrder::Lex;

    let basis_gpu = compute_groebner_basis_gpu(vec![f.clone(), g.clone()], order, None).unwrap();
    let basis_cpu = compute_groebner_basis(vec![f.clone(), g.clone()], order);

    // Mutual containment check
    for p in &basis_cpu {
        let r = cpu_reduce(p, &basis_gpu, order);
        assert!(r.is_zero(), "CPU element not in GPU basis");
    }
    for p in &basis_gpu {
        let r = cpu_reduce(p, &basis_cpu, order);
        assert!(r.is_zero(), "GPU element not in CPU basis");
    }
}

#[test]
fn agrees_with_pure_rust_f4_grevlex() {
    let f = poly(&[
        (&[1, 0, 0], 1),
        (&[0, 1, 0], 1),
        (&[0, 0, 1], 1),
        (&[0, 0, 0], -1),
    ]);
    let g = poly(&[(&[1, 0, 0], 1), (&[0, 1, 0], -1)]);
    let h = poly(&[(&[0, 1, 0], 1), (&[0, 0, 1], -1)]);
    let order = MonomialOrder::GRevLex;

    let basis_gpu =
        compute_groebner_basis_gpu(vec![f.clone(), g.clone(), h.clone()], order, None).unwrap();
    let basis_cpu = compute_groebner_basis(vec![f.clone(), g.clone(), h.clone()], order);

    for p in &basis_cpu {
        assert!(cpu_reduce(p, &basis_gpu, order).is_zero());
    }
    for p in &basis_gpu {
        assert!(cpu_reduce(p, &basis_cpu, order).is_zero());
    }
}

// ---------------------------------------------------------------------------
// MacaulayMatrix unit tests
// ---------------------------------------------------------------------------

#[test]
fn macaulay_build_and_cpu_reduce() {
    let p = 2_147_483_647u64;
    let f = poly(&[(&[2, 0], 1), (&[0, 0], -1)]);
    let g = poly(&[(&[1, 0], 1), (&[0, 0], -1)]);
    let mut mat = MacaulayMatrix::build(&[f, g], MonomialOrder::Lex, p).expect("unlucky prime");
    assert_eq!(mat.n_rows, 2);
    mat.reduce_cpu();
    // Both rows nonzero after reduction (two independent polynomials)
    let nz: Vec<bool> = (0..mat.n_rows)
        .map(|r| (0..mat.n_cols).any(|c| mat.data[r * mat.n_cols + c] != 0))
        .collect();
    assert!(nz.iter().all(|&b| b), "all rows should be nonzero: {nz:?}");
}

#[test]
fn macaulay_dependent_rows_collapse() {
    // Two identical polynomials → one row should become zero after reduction
    let p = 2_147_483_647u64;
    let f = poly(&[(&[1, 0], 1), (&[0, 0], -1)]);
    let mut mat =
        MacaulayMatrix::build(&[f.clone(), f], MonomialOrder::Lex, p).expect("unlucky prime");
    mat.reduce_cpu();
    let zero_rows = (0..mat.n_rows)
        .filter(|&r| (0..mat.n_cols).all(|c| mat.data[r * mat.n_cols + c] == 0))
        .count();
    assert_eq!(zero_rows, 1, "one of the duplicate rows must vanish");
}

// ---------------------------------------------------------------------------
// GPU path tests (requires ALKAHEST_GPU_TESTS=1 and a CUDA device)
// ---------------------------------------------------------------------------

#[test]
fn gpu_linear_system_matches_cpu() {
    if !gpu_available() {
        return;
    }
    let f = poly(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], -1)]);
    let g = poly(&[(&[1, 0], 1), (&[0, 1], -1)]);
    let order = MonomialOrder::Lex;

    let basis_gpu = compute_groebner_basis_gpu(vec![f.clone(), g.clone()], order, Some(0))
        .expect("GPU groebner failed");
    let basis_cpu = compute_groebner_basis(vec![f.clone(), g.clone()], order);

    for p in &basis_cpu {
        assert!(cpu_reduce(p, &basis_gpu, order).is_zero());
    }
    for p in &basis_gpu {
        assert!(cpu_reduce(p, &basis_cpu, order).is_zero());
    }
}

#[test]
fn gpu_circle_line_matches_cpu() {
    if !gpu_available() {
        return;
    }
    let f = poly(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -1)]);
    let g = poly(&[(&[0, 1], 1), (&[1, 0], -1)]);
    let order = MonomialOrder::GRevLex;

    let basis_gpu = compute_groebner_basis_gpu(vec![f.clone(), g.clone()], order, Some(0)).unwrap();
    let basis_cpu = compute_groebner_basis(vec![f.clone(), g.clone()], order);

    for p in &basis_cpu {
        assert!(
            cpu_reduce(p, &basis_gpu, order).is_zero(),
            "Katsura-2 CPU element not in GPU basis"
        );
    }
    for p in &basis_gpu {
        assert!(
            cpu_reduce(p, &basis_cpu, order).is_zero(),
            "Katsura-2 GPU element not in CPU basis"
        );
    }
}

#[test]
fn gpu_macaulay_reduce_kernel() {
    if !gpu_available() {
        return;
    }
    let p = 2_147_483_647u64;
    let f = poly(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -1)]);
    let g = poly(&[(&[0, 1], 1), (&[1, 0], -1)]);

    let mut mat_cpu = MacaulayMatrix::build(&[f.clone(), g.clone()], MonomialOrder::Lex, p)
        .expect("unlucky prime");
    let mut mat_gpu = MacaulayMatrix::build(&[f, g], MonomialOrder::Lex, p).expect("unlucky prime");

    mat_cpu.reduce_cpu();
    mat_gpu.reduce_gpu(0).expect("GPU reduce failed");

    assert_eq!(
        mat_cpu.data, mat_gpu.data,
        "CPU and GPU row reduction must agree mod p"
    );
}
