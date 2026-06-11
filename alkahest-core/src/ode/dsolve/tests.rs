//! Tests for the classical `dsolve` solver.  Each solving test exercises the
//! substitution verification gate implicitly (a returned solution has already
//! passed it); declines assert `Err`, never a wrong answer.

use super::*;
use crate::kernel::{Domain, ExprPool};

fn setup() -> (ExprPool, ExprId, ExprId) {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Real);
    let y = p.symbol("y", Domain::Real);
    (p, x, y)
}

/// Confirm a returned solution truly verifies (independent of the internal gate).
fn assert_verifies(input: &OdeInput, sol: &DsolveSolution, pool: &ExprPool) {
    residual_is_zero(input, sol.y_of_x, &sol.constants, pool)
        .unwrap_or_else(|e| panic!("returned solution failed verification: {e}"));
}

// ---------------------------------------------------------------------------
// First order
// ---------------------------------------------------------------------------

#[test]
fn separable_logistic() {
    // y' = y(1 - y)
    let (p, x, y) = setup();
    let (input, yp) = OdeInput::first_order(x, y, &p);
    let one = p.integer(1_i32);
    let one_minus_y = p.add(vec![one, p.mul(vec![p.integer(-1_i32), y])]);
    let rhs = p.mul(vec![y, one_minus_y]);
    // equation: y' - y(1-y) = 0
    let eq = p.add(vec![yp, p.mul(vec![p.integer(-1_i32), rhs])]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("logistic should solve");
    assert!(!res.solutions.is_empty());
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn separable_exponential() {
    // y' = y  → y = C e^x
    let (p, x, y) = setup();
    let (input, yp) = OdeInput::first_order(x, y, &p);
    let eq = p.add(vec![yp, p.mul(vec![p.integer(-1_i32), y])]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("y'=y should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn linear_first_order() {
    // y' - 3y = x   →  y' = 3y + x
    let (p, x, y) = setup();
    let (input, yp) = OdeInput::first_order(x, y, &p);
    // equation: y' - 3y - x = 0
    let eq = p.add(vec![
        yp,
        p.mul(vec![p.integer(-3_i32), y]),
        p.mul(vec![p.integer(-1_i32), x]),
    ]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("linear first order should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn bernoulli_first_order() {
    // y' + y = y^2   (n = 2)  →  equation: y' + y - y^2 = 0
    let (p, x, y) = setup();
    let (input, yp) = OdeInput::first_order(x, y, &p);
    let y2 = p.pow(y, p.integer(2_i32));
    let eq = p.add(vec![yp, y, p.mul(vec![p.integer(-1_i32), y2])]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("Bernoulli should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn exact_first_order() {
    // (2xy) dx + (x^2) dy = 0  →  y' = -2xy/x^2 = -2y/x ; exact with M=2xy, N=x^2
    // Express directly as exact via y' = -M/N with M=2x+y handled as M dx+N dy.
    // Use a genuinely exact example: (2x + y) + (x + 2y) y' = 0  (M=2x+y, N=x+2y)
    // ∂M/∂y = 1 = ∂N/∂x ✓.  F = x^2 + xy + y^2 = C.
    let (p, x, y) = setup();
    let (input, yp) = OdeInput::first_order(x, y, &p);
    let m = p.add(vec![p.mul(vec![p.integer(2_i32), x]), y]); // 2x + y
    let n = p.add(vec![x, p.mul(vec![p.integer(2_i32), y])]); // x + 2y
                                                              // equation: M + N y' = 0
    let eq = p.add(vec![m, p.mul(vec![n, yp])]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("exact should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn homogeneous_first_order() {
    // y' = (x + y)/x = 1 + y/x  (homogeneous deg 0) → y = x(C + log x)
    let (p, x, y) = setup();
    let (input, yp) = OdeInput::first_order(x, y, &p);
    let rhs = p.add(vec![p.integer(1_i32), div(y, x, &p)]);
    let eq = p.add(vec![yp, p.mul(vec![p.integer(-1_i32), rhs])]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("homogeneous should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn clairaut_first_order() {
    // y = x y' + (y')^2.  General solution y = C x + C^2.
    let (p, x, y) = setup();
    let (input, yp) = OdeInput::first_order(x, y, &p);
    let yp2 = p.pow(yp, p.integer(2_i32));
    // equation: y - x y' - (y')^2 = 0
    let eq = p.add(vec![
        y,
        p.mul(vec![p.integer(-1_i32), x, yp]),
        p.mul(vec![p.integer(-1_i32), yp2]),
    ]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("Clairaut should solve");
    assert_eq!(res.solutions[0].method, "clairaut");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn riccati_with_polynomial_particular() {
    // y' = y^2 - x^2 + 1 has particular solution y_p = x.
    // (y_p' = 1 = x^2 - x^2 + 1 ✓)
    let (p, x, y) = setup();
    let (input, yp) = OdeInput::first_order(x, y, &p);
    let y2 = p.pow(y, p.integer(2_i32));
    let x2 = p.pow(x, p.integer(2_i32));
    let rhs = p.add(vec![
        y2,
        p.mul(vec![p.integer(-1_i32), x2]),
        p.integer(1_i32),
    ]);
    let eq = p.add(vec![yp, p.mul(vec![p.integer(-1_i32), rhs])]);
    let input = input.with_equation(eq);
    match dsolve(&input, &p) {
        Ok(res) => assert_verifies(&input, &res.solutions[0], &p),
        // Acceptable to decline if the linear reduction integral does not close,
        // but it must never return a wrong answer.
        Err(DsolveError::Unsupported(_)) => {}
        Err(e) => panic!("unexpected error: {e}"),
    }
}

#[test]
fn riccati_declined_without_particular() {
    // A Riccati with no low-degree polynomial particular solution must decline.
    // y' = y^2 + x  (no polynomial particular solution of degree ≤ 2)
    let (p, x, y) = setup();
    let (input, yp) = OdeInput::first_order(x, y, &p);
    let y2 = p.pow(y, p.integer(2_i32));
    let rhs = p.add(vec![y2, x]);
    let eq = p.add(vec![yp, p.mul(vec![p.integer(-1_i32), rhs])]);
    let input = input.with_equation(eq);
    assert!(
        dsolve(&input, &p).is_err(),
        "should decline Riccati w/o particular"
    );
}

// ---------------------------------------------------------------------------
// Second order constant coefficient
// ---------------------------------------------------------------------------

#[test]
fn harmonic_oscillator() {
    // y'' + y = 0  → y = C1 cos x + C2 sin x
    let (p, x, y) = setup();
    let (input, _yp, ypp) = OdeInput::second_order(x, y, &p);
    let eq = p.add(vec![ypp, y]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("harmonic oscillator should solve");
    assert_eq!(res.solutions[0].constants.len(), 2);
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn real_distinct_roots() {
    // y'' - 3y' + 2y = 0  → roots 1,2 → y = C1 e^x + C2 e^{2x}
    let (p, x, y) = setup();
    let (input, yp, ypp) = OdeInput::second_order(x, y, &p);
    let eq = p.add(vec![
        ypp,
        p.mul(vec![p.integer(-3_i32), yp]),
        p.mul(vec![p.integer(2_i32), y]),
    ]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("distinct roots should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn repeated_root() {
    // y'' - 2y' + y = 0  → double root 1 → y = (C1 + C2 x) e^x
    let (p, x, y) = setup();
    let (input, yp, ypp) = OdeInput::second_order(x, y, &p);
    let eq = p.add(vec![ypp, p.mul(vec![p.integer(-2_i32), yp]), y]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("repeated root should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn complex_roots() {
    // y'' + 2y' + 5y = 0  → roots -1 ± 2i
    let (p, x, y) = setup();
    let (input, yp, ypp) = OdeInput::second_order(x, y, &p);
    let eq = p.add(vec![
        ypp,
        p.mul(vec![p.integer(2_i32), yp]),
        p.mul(vec![p.integer(5_i32), y]),
    ]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("complex roots should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn undetermined_coefficients_x_exp_x() {
    // y'' - y = x e^x.  RHS = x·e^x (resonance: e^x is homogeneous).
    let (p, x, y) = setup();
    let (input, _yp, ypp) = OdeInput::second_order(x, y, &p);
    let xex = p.mul(vec![x, p.func("exp", vec![x])]);
    // equation: y'' - y - x e^x = 0
    let eq = p.add(vec![
        ypp,
        p.mul(vec![p.integer(-1_i32), y]),
        p.mul(vec![p.integer(-1_i32), xex]),
    ]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("undetermined coefficients should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn variation_of_parameters_tan() {
    // y'' + y = tan(x).  Variation of parameters; integrate may or may not close.
    let (p, x, y) = setup();
    let (input, _yp, ypp) = OdeInput::second_order(x, y, &p);
    let tanx = p.func("tan", vec![x]);
    let eq = p.add(vec![ypp, y, p.mul(vec![p.integer(-1_i32), tanx])]);
    let input = input.with_equation(eq);
    match dsolve(&input, &p) {
        Ok(res) => assert_verifies(&input, &res.solutions[0], &p),
        Err(DsolveError::Unsupported(_)) => {} // acceptable decline if integral doesn't close
        Err(e) => panic!("must decline, not error wrongly: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Euler–Cauchy
// ---------------------------------------------------------------------------

#[test]
fn euler_cauchy_distinct() {
    // x^2 y'' + 2x y' - 2y = 0  → indicial m^2 + m - 2 = 0 → m = 1, -2
    let (p, x, y) = setup();
    let (input, yp, ypp) = OdeInput::second_order(x, y, &p);
    let x2 = p.pow(x, p.integer(2_i32));
    let eq = p.add(vec![
        p.mul(vec![x2, ypp]),
        p.mul(vec![p.integer(2_i32), x, yp]),
        p.mul(vec![p.integer(-2_i32), y]),
    ]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("Euler-Cauchy should solve");
    assert_eq!(res.solutions[0].method, "euler_cauchy");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn euler_cauchy_repeated() {
    // x^2 y'' - x y' + y = 0  → m^2 - 2m + 1 = 0 → double root m=1
    let (p, x, y) = setup();
    let (input, yp, ypp) = OdeInput::second_order(x, y, &p);
    let x2 = p.pow(x, p.integer(2_i32));
    let eq = p.add(vec![
        p.mul(vec![x2, ypp]),
        p.mul(vec![p.integer(-1_i32), x, yp]),
        y,
    ]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("Euler-Cauchy repeated should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

// ---------------------------------------------------------------------------
// Higher order
// ---------------------------------------------------------------------------

#[test]
fn third_order_constant_coeff() {
    // y''' - 6y'' + 11y' - 6y = 0  → roots 1,2,3
    let (p, x, y) = setup();
    let (input, derivs) = OdeInput::higher_order(x, y, 3, &p);
    let (yp, ypp, yppp) = (derivs[0], derivs[1], derivs[2]);
    let eq = p.add(vec![
        yppp,
        p.mul(vec![p.integer(-6_i32), ypp]),
        p.mul(vec![p.integer(11_i32), yp]),
        p.mul(vec![p.integer(-6_i32), y]),
    ]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("third order should solve");
    assert_eq!(res.solutions[0].constants.len(), 3);
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn fresh_constants_avoid_user_symbols() {
    // If the equation mentions a symbol "C1", the generator must skip it.
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Real);
    let y = p.symbol("y", Domain::Real);
    let c1 = p.symbol("C1", Domain::Real);
    let (input, _yp, ypp) = OdeInput::second_order(x, y, &p);
    // y'' + y = 0 but with a stray C1 multiplier in a vanishing term so it is
    // recorded as used.  (C1 - C1) * x adds zero but registers the name.
    let zero_term = p.mul(vec![p.add(vec![c1, p.mul(vec![p.integer(-1_i32), c1])]), x]);
    let eq = p.add(vec![ypp, y, zero_term]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("should still solve");
    for c in &res.solutions[0].constants {
        assert_ne!(*c, c1, "fresh constant collided with user symbol C1");
    }
}
