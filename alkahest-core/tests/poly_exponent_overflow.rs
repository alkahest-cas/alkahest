//! Exponents past `u32` must be refused or computed exactly — never wrapped.
//!
//! Every sparse polynomial type keys its terms by `u32` exponents, and the
//! arithmetic that built them used plain `+`: in a release build
//! `x^(2^31)·x^(2^31)` became `x^0 = 1`, so `solve` reported no solutions,
//! `horner`, `poly_normal` and `real_roots` answered for a different
//! polynomial, and a resultant of degree 6·10⁹ came back reduced mod 2^32.
//! These tests use only the public API, and every one of them failed on the
//! commit before the fix (the resultant one with a wrong answer, the rest by
//! returning `Ok` for a polynomial that is not the input).

use alkahest_cas::kernel::{Domain, ExprId, ExprPool};
use alkahest_cas::poly::{poly_normal, resultant, ConversionError, MultiPoly, UniPoly};
use alkahest_cas::solver::{expr_to_gbpoly, solve_polynomial_system};

const B31: u64 = 1 << 31;

fn setup() -> (ExprPool, ExprId, ExprId) {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Real);
    let y = p.symbol("y", Domain::Real);
    (p, x, y)
}

fn pw(p: &ExprPool, b: ExprId, e: u64) -> ExprId {
    p.pow(b, p.integer(e))
}

/// `x^(2^31) · x^(2^31)`, kept as an unevaluated product.
fn w(p: &ExprPool, x: ExprId) -> ExprId {
    p.mul(vec![pw(p, x, B31), pw(p, x, B31)])
}

fn minus(p: &ExprPool, e: ExprId) -> ExprId {
    p.mul(vec![p.integer(-1_i32), e])
}

#[test]
fn multipoly_product_exponent_is_refused() {
    let (p, x, y) = setup();
    assert_eq!(
        MultiPoly::from_symbolic(w(&p, x), vec![x, y], &p),
        Err(ConversionError::ExponentTooLarge)
    );
    // (x^65536)^65536
    let nested = p.pow(pw(&p, x, 65536), p.integer(65536_u32));
    assert_eq!(
        MultiPoly::from_symbolic(nested, vec![x], &p),
        Err(ConversionError::ExponentTooLarge)
    );
    // factor_z(x^(2^32) - y^2) used to factor 1 - y^2.
    let f = p.add(vec![w(&p, x), minus(&p, pw(&p, y, 2))]);
    assert_eq!(
        MultiPoly::from_symbolic(f, vec![x, y], &p),
        Err(ConversionError::ExponentTooLarge)
    );
}

#[test]
fn multipoly_total_degree_past_u32_is_refused() {
    // total_degree(x^(2^31) y^(2^31)) used to be 0.
    let (p, x, y) = setup();
    let e = p.mul(vec![pw(&p, x, B31), pw(&p, y, B31)]);
    assert_eq!(
        MultiPoly::from_symbolic(e, vec![x, y], &p).map(|m| m.total_degree()),
        Err(ConversionError::ExponentTooLarge)
    );
}

#[test]
fn unipoly_product_degree_is_refused() {
    // real_roots(x^(2^32) - 4) and horner(x^(2^32) + x) read a wrapped degree.
    let (p, x, _) = setup();
    let f = p.add(vec![w(&p, x), p.integer(-4_i32)]);
    assert_eq!(
        UniPoly::from_symbolic(f, x, &p).err(),
        Some(ConversionError::ExponentTooLarge)
    );
    let g = p.add(vec![w(&p, x), x]);
    assert_eq!(
        alkahest_cas::horner::horner(g, x, &p),
        Err(ConversionError::ExponentTooLarge)
    );
}

#[test]
fn poly_normal_is_refused() {
    // poly_normal(x^(2^32)·y - y) used to be 0.
    let (p, x, y) = setup();
    let e = p.add(vec![p.mul(vec![w(&p, x), y]), minus(&p, y)]);
    assert_eq!(
        poly_normal(e, vec![x, y], &p),
        Err(ConversionError::ExponentTooLarge)
    );
}

#[test]
fn resultant_past_u32_is_exact() {
    // res(x² - y^N, x - y^N, x) = y^(2N) - y^N; N = 3·10⁹ came back as
    // -y^3000000000 + y^1705032704.
    let (p, x, y) = setup();
    let n = 3_000_000_000_u64;
    let yn = pw(&p, y, n);
    let f = p.add(vec![pw(&p, x, 2), minus(&p, yn)]);
    let g = p.add(vec![x, minus(&p, yn)]);
    let r = resultant(f, g, x, &p).unwrap().value;
    let want = p.add(vec![minus(&p, yn), pw(&p, y, 2 * n)]);
    assert_eq!(r, want, "got {}", p.display(r));
}

#[test]
fn groebner_conversion_and_solve_refuse() {
    // solve([x^(2^32) - 2, y - x]) used to report no solutions.
    let (p, x, y) = setup();
    let f = p.add(vec![w(&p, x), p.integer(-2_i32)]);
    assert!(expr_to_gbpoly(f, &[x, y], &p).is_err());
    let g = p.add(vec![y, minus(&p, x)]);
    let r = solve_polynomial_system(vec![f, g], vec![x, y], &p);
    assert!(r.is_err(), "solve accepted x^(2^32) - 2");
    // x^(2^31) · y^(2^31): each exponent fits, the total degree does not.
    let h = p.mul(vec![pw(&p, x, B31), pw(&p, y, B31)]);
    assert!(expr_to_gbpoly(h, &[x, y], &p).is_err());
    // x^(2^31) itself is fine: the power loop no longer squares past it.
    let k = expr_to_gbpoly(pw(&p, x, B31), &[x, y], &p).unwrap();
    assert!(k.terms.contains_key(&vec![1 << 31, 0]));
}

#[test]
fn exponent_boundaries() {
    let (p, x, y) = setup();
    let top = MultiPoly::from_symbolic(pw(&p, x, u64::from(u32::MAX)), vec![x, y], &p).unwrap();
    assert_eq!(top.total_degree(), u32::MAX);
    for e in [1_u64 << 32, 1 << 63, u64::MAX] {
        assert_eq!(
            MultiPoly::from_symbolic(pw(&p, x, e), vec![x, y], &p),
            Err(ConversionError::ExponentTooLarge),
            "exponent {e}"
        );
        assert!(expr_to_gbpoly(pw(&p, x, e), &[x, y], &p).is_err());
    }
}
