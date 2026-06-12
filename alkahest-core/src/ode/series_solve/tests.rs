//! Tests for series / Frobenius ODE solutions.

use super::*;
use crate::kernel::{Domain, ExprPool};
use rug::Rational;

fn ri(n: i64) -> Rational {
    Rational::from(n)
}
fn rr(n: i64, d: i64) -> Rational {
    Rational::from((n, d))
}

/// Build the ODE p·y''+q·y'+r·y from coefficient builders and solve about x₀.
fn solve(
    pool: &ExprPool,
    x: ExprId,
    p: ExprId,
    q: ExprId,
    r: ExprId,
    x0: ExprId,
    order: usize,
) -> SeriesResult {
    let ode = SeriesOde::new(x, p, q, r);
    series_solve(&ode, x0, order, pool).expect("series_solve should succeed")
}

// ---------------------------------------------------------------------------
// Ordinary point: y'' + y = 0 → cos x, sin x.
// ---------------------------------------------------------------------------
#[test]
fn ordinary_y_pp_plus_y() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let one = pool.integer(1);
    let zero = pool.integer(0);
    let res = solve(&pool, x, one, zero, one, zero, 9);
    assert_eq!(res.kind, PointKind::Ordinary);
    assert_eq!(res.solutions.len(), 2);
    // a0=1,a1=0 → cos x = 1 - x²/2 + x⁴/24 - x⁶/720 + …
    let c = &res.solutions[0].coeffs;
    assert_eq!(c[0], ri(1));
    assert_eq!(c[1], ri(0));
    assert_eq!(c[2], rr(-1, 2));
    assert_eq!(c[3], ri(0));
    assert_eq!(c[4], rr(1, 24));
    assert_eq!(c[6], rr(-1, 720));
    // a0=0,a1=1 → sin x = x - x³/6 + x⁵/120 - …
    let s = &res.solutions[1].coeffs;
    assert_eq!(s[0], ri(0));
    assert_eq!(s[1], ri(1));
    assert_eq!(s[3], rr(-1, 6));
    assert_eq!(s[5], rr(1, 120));
}

// ---------------------------------------------------------------------------
// Airy: y'' − x y = 0 about 0. Ordinary point.
// First solution a0=1,a1=0: 1 + x³/6 + x⁶/180 + …
// Recurrence a_{n+2} = a_{n-1}/((n+2)(n+1)).
// ---------------------------------------------------------------------------
#[test]
fn airy_ordinary() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let one = pool.integer(1);
    let zero = pool.integer(0);
    let neg_x = pool.mul(vec![pool.integer(-1), x]);
    let res = solve(&pool, x, one, zero, neg_x, zero, 12);
    assert_eq!(res.kind, PointKind::Ordinary);
    let c = &res.solutions[0].coeffs; // a0=1,a1=0
    assert_eq!(c[0], ri(1));
    assert_eq!(c[1], ri(0));
    assert_eq!(c[2], ri(0));
    assert_eq!(c[3], rr(1, 6)); // 1/(3·2)
    assert_eq!(c[6], rr(1, 180)); // 1/(6·5)·1/6 = 1/180
                                  // second solution a0=0,a1=1: x + x⁴/12 + …
    let s = &res.solutions[1].coeffs;
    assert_eq!(s[1], ri(1));
    assert_eq!(s[4], rr(1, 12)); // 1/(4·3)
}

// ---------------------------------------------------------------------------
// Legendre l=2: (1−x²)y'' − 2x y' + 6 y = 0. Even solution terminates: 1 − 3x².
// ---------------------------------------------------------------------------
#[test]
fn legendre_l2_polynomial() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let one = pool.integer(1);
    let x2 = pool.pow(x, pool.integer(2));
    // p = 1 − x²
    let p = pool.add(vec![one, pool.mul(vec![pool.integer(-1), x2])]);
    // q = −2x
    let q = pool.mul(vec![pool.integer(-2), x]);
    // r = 6
    let r = pool.integer(6);
    let zero = pool.integer(0);
    let res = solve(&pool, x, p, q, r, zero, 10);
    assert_eq!(res.kind, PointKind::Ordinary);
    // even solution a0=1,a1=0 → 1 − 3x², all higher vanish.
    let c = &res.solutions[0].coeffs;
    assert_eq!(c[0], ri(1));
    assert_eq!(c[2], ri(-3));
    for (k, ck) in c.iter().enumerate().skip(3) {
        assert_eq!(
            *ck,
            ri(0),
            "coeff {k} should vanish (polynomial terminates)"
        );
    }
}

// ---------------------------------------------------------------------------
// Bessel J₀: x²y'' + x y' + x² y = 0 about 0. Regular singular, indicial r²=0.
// First solution 1 − x²/4 + x⁴/64 − x⁶/2304 + …
// ---------------------------------------------------------------------------
#[test]
fn bessel_j0_frobenius() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let x2 = pool.pow(x, pool.integer(2));
    let zero = pool.integer(0);
    let res = solve(&pool, x, x2, x, x2, zero, 9);
    assert_eq!(res.kind, PointKind::RegularSingular);
    let s1 = &res.solutions[0];
    assert_eq!(s1.exponent, ri(0));
    let c = &s1.coeffs;
    assert_eq!(c[0], ri(1));
    assert_eq!(c[2], rr(-1, 4));
    assert_eq!(c[4], rr(1, 64));
    assert_eq!(c[6], rr(-1, 2304));
    // Bessel J0 known: a_{2m} = (-1)^m / (4^m (m!)²).
    // m=1: -1/4, m=2: 1/64, m=3: -1/2304. ✓
}

// ---------------------------------------------------------------------------
// Equal-root log second solution for Bessel J₀ exists (Y₀-style log term).
// ---------------------------------------------------------------------------
#[test]
fn bessel_j0_has_log_second_solution() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let x2 = pool.pow(x, pool.integer(2));
    let res = solve(&pool, x, x2, x, x2, pool.integer(0), 8);
    assert_eq!(
        res.solutions.len(),
        2,
        "log second solution should be produced"
    );
    let s2 = &res.solutions[1];
    assert!(
        s2.log_coeff.is_some(),
        "second solution should carry a log term"
    );
    assert_eq!(s2.log_coeff.clone().unwrap(), ri(1));
}

// ---------------------------------------------------------------------------
// Non-integer root gap (case a): Euler-like x²y'' + x y' − (1/4) y = 0.
// Indicial r² − 1/4 = 0 → r = ±1/2. Gap = 1 ∈ ℤ actually... pick distinct.
// Use x²y'' + x·(1/2)y' ... → choose roots ±1/3 (gap 2/3 non-integer).
// Indicial r² + (P0−1)r + Q0 with P0=p, Q0=q.  Want roots 1/3, −1/3:
//   r²+0·r −1/9 = 0 → P0=1, Q0=−1/9. So tq/p|0 = 1 → q = (1/x)·p·? Build
//   p=x², q=x, r=−1/9. Then P0 = (t·q/p)|0 = (x·x/x²)=1, Q0=(x²·r/x²)=−1/9. ✓
// ---------------------------------------------------------------------------
#[test]
fn frobenius_noninteger_gap_two_series() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let x2 = pool.pow(x, pool.integer(2));
    let r = pool.rational(-1, 9);
    let res = solve(&pool, x, x2, x, r, pool.integer(0), 6);
    assert_eq!(res.kind, PointKind::RegularSingular);
    assert_eq!(res.solutions.len(), 2);
    assert_eq!(res.solutions[0].exponent, rr(1, 3));
    assert_eq!(res.solutions[1].exponent, rr(-1, 3));
    assert!(res.solutions[0].log_coeff.is_none());
    assert!(res.solutions[1].log_coeff.is_none());
}

// ---------------------------------------------------------------------------
// Euler equation x²y'' − x y' + y = 0: indicial (r−1)² = 0 (equal root r=1).
// Solutions x and x·ln x. Frobenius bracket for y₁ is just [1] (b_n=0 for n≥1),
// and the log second solution y₂ = x ln x + (bracket all zero).
// ---------------------------------------------------------------------------
#[test]
fn euler_equal_root_log() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let x2 = pool.pow(x, pool.integer(2));
    let neg_x = pool.mul(vec![pool.integer(-1), x]);
    let one = pool.integer(1);
    let res = solve(&pool, x, x2, neg_x, one, pool.integer(0), 6);
    assert_eq!(res.kind, PointKind::RegularSingular);
    assert_eq!(res.solutions.len(), 2);
    assert_eq!(res.solutions[0].exponent, ri(1));
    // y₁ = x (bracket [1,0,0,…]).
    assert_eq!(res.solutions[0].coeffs[0], ri(1));
    for k in 1..res.solutions[0].coeffs.len() {
        assert_eq!(res.solutions[0].coeffs[k], ri(0));
    }
    // y₂ = x·ln x  → log_coeff = 1, bracket all zero.
    let s2 = &res.solutions[1];
    assert_eq!(s2.exponent, ri(1));
    assert_eq!(s2.log_coeff.clone().unwrap(), ri(1));
    for k in 0..s2.coeffs.len() {
        assert_eq!(
            s2.coeffs[k],
            ri(0),
            "Euler log bracket should be zero at {k}"
        );
    }
}

// ---------------------------------------------------------------------------
// to_expr renders something nonempty for the ordinary case.
// ---------------------------------------------------------------------------
#[test]
fn to_expr_renders() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let one = pool.integer(1);
    let zero = pool.integer(0);
    let res = solve(&pool, x, one, zero, one, zero, 7);
    let e = res.solutions[0].to_expr(x, zero, 7, &pool);
    let s = pool.display(e).to_string();
    assert!(s.contains('x'), "rendered expr should mention x: {s}");
}

// ---------------------------------------------------------------------------
// Irregular singular point declines: y'' + (1/x²) y = 0 has q analytic but
// r/p = 1/x² → t²·r/p = 1 ok actually regular. Use a genuinely irregular one:
//   x³ y'' + y = 0: p=x³, r=1. t·q/p with q=0 ok; t²·r/p = t²/x³ = 1/x → pole.
// ---------------------------------------------------------------------------
#[test]
fn irregular_singular_declines() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let x3 = pool.pow(x, pool.integer(3));
    let one = pool.integer(1);
    let zero = pool.integer(0);
    let ode = SeriesOde::new(x, x3, zero, one);
    let res = series_solve(&ode, pool.integer(0), 6, &pool);
    assert!(
        matches!(
            res,
            Err(SeriesError::IrregularSingular) | Err(SeriesError::NotAnalytic(_))
        ),
        "expected irregular/non-analytic decline, got {res:?}"
    );
}

// ---------------------------------------------------------------------------
// Positive-integer root gap with log (Bessel J₁): x²y'' + x y' + (x²−1)y = 0.
// Indicial r²−1=0 → roots ±1 (gap 2). Larger-root solution y₁ = t·∑…; the
// second solution carries a log term. Exercises frobenius_log_integer + gate.
// ---------------------------------------------------------------------------
#[test]
fn bessel_j1_integer_gap_log() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let x2 = pool.pow(x, pool.integer(2));
    // r = x² − 1
    let r = pool.add(vec![x2, pool.integer(-1)]);
    let res = solve(&pool, x, x2, x, r, pool.integer(0), 8);
    assert_eq!(res.kind, PointKind::RegularSingular);
    // Larger root r₁ = 1.
    assert_eq!(res.solutions[0].exponent, ri(1));
    assert!(res.solutions[0].log_coeff.is_none());
    // y₁ (Bessel J₁ shape): b₀=1, b₂=−1/8, b₄=1/192, … (bracket index = power).
    let c = &res.solutions[0].coeffs;
    assert_eq!(c[0], ri(1));
    assert_eq!(c[2], rr(-1, 8));
    assert_eq!(c[4], rr(1, 192));
    // If a second (log) solution is returned it is exponent −1 and gated; the
    // construction may honestly decline, in which case only y₁ is present.
    if res.solutions.len() == 2 {
        let s2 = &res.solutions[1];
        assert_eq!(s2.exponent, ri(-1));
        assert!(
            s2.log_coeff.is_some(),
            "integer-gap Bessel J₁ needs a log term"
        );
    } else {
        assert!(res.second_solution_declined());
    }
}
