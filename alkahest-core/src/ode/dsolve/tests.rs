//! Tests for the classical `dsolve` solver.  Each solving test exercises the
//! substitution verification gate implicitly (a returned solution has already
//! passed it); declines assert `Err`, never a wrong answer.

use super::*;
use crate::integrate::special::basis_functions_used;
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

#[test]
fn nonhomogeneous_polynomial_rhs() {
    // y'' - y = x^2 + 1.  Undetermined coefficients (polynomial RHS).
    let (p, x, y) = setup();
    let (input, _yp, ypp) = OdeInput::second_order(x, y, &p);
    let rhs = p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(1_i32)]);
    let eq = p.add(vec![
        ypp,
        p.mul(vec![p.integer(-1_i32), y]),
        p.mul(vec![p.integer(-1_i32), rhs]),
    ]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("polynomial RHS should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn nonhomogeneous_nonresonant_exp() {
    // y'' - y = e^{2x}.  Undetermined coefficients (non-resonant exp).
    let (p, x, y) = setup();
    let (input, _yp, ypp) = OdeInput::second_order(x, y, &p);
    let e2x = p.func("exp", vec![p.mul(vec![p.integer(2_i32), x])]);
    let eq = p.add(vec![
        ypp,
        p.mul(vec![p.integer(-1_i32), y]),
        p.mul(vec![p.integer(-1_i32), e2x]),
    ]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("non-resonant exp RHS should solve");
    assert_verifies(&input, &res.solutions[0], &p);
}

#[test]
fn fourth_order_constant_coeff() {
    // y'''' - y = 0  → roots ±1, ±i → e^x, e^{-x}, cos x, sin x
    let (p, x, y) = setup();
    let (input, derivs) = OdeInput::higher_order(x, y, 4, &p);
    let eq = p.add(vec![derivs[3], p.mul(vec![p.integer(-1_i32), y])]);
    let input = input.with_equation(eq);
    let res = dsolve(&input, &p).expect("fourth order should solve");
    assert_eq!(res.solutions[0].constants.len(), 4);
    assert_verifies(&input, &res.solutions[0], &p);
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

// ---------------------------------------------------------------------------
// Variation of parameters, Euler–Cauchy forcing, reduction of order
//
// Each case is stated in source form and solved through the normal entry
// point; the solution is then re-substituted into the *original* equation,
// independently of the gate inside `dsolve`.
// ---------------------------------------------------------------------------

/// Solve `src = 0` (in `x`, `y`, `yp`, `ypp`, …) and return the pool, input
/// and the first solution, asserting it verifies by substitution.
fn solve_src(order: usize, src: &str) -> (ExprPool, OdeInput, DsolveSolution) {
    let pool = ExprPool::new();
    let input = super::corpus::build_ode(order, src, &pool).expect("corpus source parses");
    let res = dsolve(&input, &pool).unwrap_or_else(|e| panic!("`{src}` should solve: {e}"));
    let sol = res.solutions[0].clone();
    residual_is_zero(&input, sol.y_of_x, &sol.constants, &pool)
        .unwrap_or_else(|e| panic!("`{src}` returned an unverified solution: {e}"));
    (pool, input, sol)
}

/// Assert `src` declines cleanly — an `Unsupported`, never a wrong answer and
/// never an unevaluated integral.
fn assert_declines(order: usize, src: &str) {
    let pool = ExprPool::new();
    let input = super::corpus::build_ode(order, src, &pool).expect("corpus source parses");
    match dsolve(&input, &pool) {
        Err(DsolveError::Unsupported(_)) => {}
        Err(e) => panic!("`{src}` should decline as Unsupported, got {e}"),
        Ok(res) => panic!(
            "`{src}` should decline, but returned {}",
            pool.display(res.solutions[0].y_of_x)
        ),
    }
}

#[test]
fn vop_forcing_no_ansatz_can_express() {
    // Forcing terms outside every undetermined-coefficients ansatz
    // (poly × exp × sin/cos).  These reach a closed form only through
    // variation of parameters.
    for src in [
        "ypp + y - 1/cos(x)",        // sec x
        "ypp + y - tan(x)",          // tan x
        "ypp + y - 1/sin(x)",        // csc x
        "ypp + y - tan(x)/cos(x)",   // sec x tan x
        "ypp + y - 1/(1 + sin(x))",  // rational in sin
        "ypp - 2*yp + y - exp(x)/x", // e^x/x with a resonant basis
        "ypp - y - 1/(1 + exp(x))",  // logistic forcing
        "ypp - y - exp(x)/(1 + exp(x))",
    ] {
        let (_, _, sol) = solve_src(2, src);
        assert_eq!(sol.constants.len(), 2, "`{src}`: wrong constant count");
    }
}

#[test]
fn vop_generalises_above_second_order() {
    // Third order with a forcing no ansatz covers: the Wronskian is 3×3 and
    // the particular solution needs Cramer's rule, not the 2×2 formula.
    for src in ["yppp + yp - 1/cos(x)", "yppp + yp - tan(x)"] {
        let (_, _, sol) = solve_src(3, src);
        assert_eq!(sol.constants.len(), 3, "`{src}`: wrong constant count");
    }
}

#[test]
fn euler_cauchy_nonhomogeneous() {
    // Euler–Cauchy with forcing: basis {x^m₁, x^m₂} then variation of
    // parameters.  Previously declined for every non-zero right-hand side.
    for src in [
        "x^2*ypp + x*yp - y - x^2",
        "x^2*ypp - 2*x*yp + 2*y - x^3",
        "x^2*ypp + x*yp - y - log(x)",
        "x^2*ypp - x*yp + y - x",
    ] {
        let (_, _, sol) = solve_src(2, src);
        assert_eq!(sol.method, "euler_cauchy");
        assert_eq!(sol.constants.len(), 2, "`{src}`: wrong constant count");
    }
}

#[test]
fn reduction_of_order_variable_coefficients() {
    // Second order with genuinely variable coefficients — neither
    // constant-coefficient nor Euler–Cauchy.  One solution by ansatz, the
    // second by y₂ = y₁∫e^{−∫P}/y₁², the forcing by variation of parameters.
    for src in [
        "ypp - (2/x)*yp + (2/x^2)*y",   // y₁ = x
        "(1 - x^2)*ypp - 2*x*yp + 2*y", // Legendre, y₁ = x
        "x*ypp - (x + 1)*yp + y",       // y₁ = e^x
        "ypp - yp/x - x",               // y₁ = 1, forced
    ] {
        let (_, _, sol) = solve_src(2, src);
        assert_eq!(sol.method, "reduction_of_order");
        assert_eq!(sol.constants.len(), 2, "`{src}`: wrong constant count");
    }
}

#[test]
fn integrating_factor_folds_logarithms() {
    // μ = e^{∫p dx} used to be emitted as a literal `exp(−log x)`, and the
    // integration engine was then asked for ∫q·e^{−log x} dx.  These all need
    // the fold to reach an elementary integrand.
    for src in [
        "yp - y/x - x*log(x)",
        "yp + y*tan(x) - sin(x)",
        "yp + y/(x*log(x)) - 1",
        "yp + y/x - cos(x)/x",
        "yp - 2*x*y - exp(x^2)*sin(x)",
        "yp + y - x*exp(-x)",
    ] {
        let (_, _, sol) = solve_src(1, src);
        assert_eq!(sol.constants.len(), 1, "`{src}`: wrong constant count");
    }
}

#[test]
fn quadrature_over_the_special_function_basis_closes() {
    // These declined until the special-function emitters became reachable from
    // the integrator: variation of parameters needs `∫ e^{∓x}/x dx` for the
    // first and `∫ sin(x)/x dx`, `∫ cos(x)/x dx` for the second, none of which
    // is elementary.  They are all in `integrate::special::SPECIAL_BASIS`, so
    // the answer is `Ei` for the first and `Si`/`Ci` for the second.
    //
    // Solving is only half of it: `dsolve` found these answers before and its
    // gate threw them away, because the residual cancels only over *rational*
    // coefficients (`1/(2x) − 1/(2x)`), which `collect_add_terms` could not do
    // until it carried `rug::Rational`.  `solve_src` re-substitutes into the
    // original equation independently of that gate, and `verify::eval_func`
    // has no `f64` kernel for `Ei`/`Si`/`Ci` — so its numeric fallback cannot
    // fire and these pass on an exact symbolic `residual ≡ 0`.
    for (src, basis) in [
        ("ypp - y - 1/x", &["Ei"] as &[&str]),
        // `basis_functions_used` returns the names sorted.
        ("ypp + y - 1/x", &["Ci", "Si"]),
    ] {
        let (pool, _, sol) = solve_src(2, src);
        assert_eq!(sol.constants.len(), 2, "`{src}`: wrong constant count");
        assert_eq!(
            basis_functions_used(sol.y_of_x, &pool),
            basis,
            "`{src}`: wrong special-function vocabulary in {}",
            pool.display(sol.y_of_x)
        );
        for c in &sol.constants {
            assert!(
                super::contains(sol.y_of_x, *c, &pool),
                "`{src}`: constant {} does not appear in the solution",
                pool.display(*c)
            );
        }
    }
}

#[test]
fn declines_when_the_quadrature_leaves_the_special_function_basis() {
    // `y'' + y = x/(1+x²)`.  Variation of parameters asks for
    // `∫ x·sin(x)/(1+x²) dx` and `∫ x·cos(x)/(1+x²) dx`.  Neither is
    // elementary, and — unlike `∫ sin(x)/x dx` above — neither is anything
    // Alkahest can *name*.  Over ℝ the denominator is irreducible; over ℂ,
    // `x/(1+x²) = ½·[1/(x−i) + 1/(x+i)]`, so the closed form is `Si`/`Ci` at
    // the complex arguments `x ± i`.  Two things put that out of reach:
    //
    //   * every kernel in `primitive::expint` is real-argument only and
    //     refuses (`None`) off the real axis, on purpose — see its module
    //     docs; there is no complex-argument `Ei`/`Si`/`Ci` primitive to emit.
    //   * `integrate::special`'s emitter table covers `c·f(g)/d` only for `g`
    //     and `d` *linear* with `d ∝ g`, so a quadratic denominator never
    //     matches, whatever the arguments.
    //
    // So the honest answer is a decline — an `Unsupported`, not an
    // unevaluated integral and not a claim about the ODE.  (`integrate` may
    // legitimately certify the *integral* non-elementary; what must never
    // happen is that verdict being re-read as a statement about `dsolve`.)
    //
    // **This test's premise expires** the moment either bullet above stops
    // holding: a complex-argument expint primitive, or a partial-fraction
    // route in `integrate::special` that handles `f(g)/q` for `deg q > 1`.
    // If it then fails with a *solved* equation rather than a decline, that is
    // the premise expiring and not a regression — check the returned solution
    // by differentiation, move this source up into
    // `quadrature_over_the_special_function_basis_closes`, and pick a fresh
    // integrand for this test.  The decline path is what is worth keeping
    // here; the integrand is only the current way of reaching it.
    assert_declines(2, "ypp + y - x/(1 + x^2)");
}

#[test]
fn general_solution_has_exactly_order_many_constants() {
    // Variation of parameters must not allocate a constant of its own: the
    // particular solution contributes none, so the count is the order.
    for (order, src) in [
        (2_usize, "ypp + y - 1/cos(x)"),
        (2, "x^2*ypp + x*yp - y - log(x)"),
        (2, "ypp - yp/x - x"),
        (3, "yppp + yp - tan(x)"),
    ] {
        let (pool, _, sol) = solve_src(order, src);
        assert_eq!(sol.constants.len(), order, "`{src}`");
        for c in &sol.constants {
            assert!(
                super::contains(sol.y_of_x, *c, &pool),
                "`{src}`: constant {} does not appear in the solution",
                pool.display(*c)
            );
        }
    }
}
