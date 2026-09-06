//! Tests for the classical `dsolve` solver.  Each solving test exercises the
//! substitution verification gate implicitly (a returned solution has already
//! passed it); declines assert `Err`, never a wrong answer.

use super::*;
use crate::deriv::SideCondition;
use crate::integrate::special::basis_functions_used;
use crate::kernel::{Domain, ExprPool};
use crate::simplify::assumptions::AssumptionContext;

fn setup() -> (ExprPool, ExprId, ExprId) {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Real);
    let y = p.symbol("y", Domain::Real);
    (p, x, y)
}

/// Confirm a returned solution truly verifies (independent of the internal gate).
fn assert_verifies(input: &OdeInput, sol: &DsolveSolution, pool: &ExprPool) {
    solution_is_verified(input, sol, pool)
        .unwrap_or_else(|e| panic!("returned solution failed verification: {e}"));
}

/// The explicit `y(x)` of a solution that must be explicit.
fn explicit_of(sol: &DsolveSolution) -> ExprId {
    sol.y_of_x()
        .expect("expected an explicit y(x), got an implicit relation")
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
        // but it must never return a wrong answer — and the decline has to say
        // *which* integral, which here is the genuinely non-elementary
        // `∫ e^{x²} dx` the `y = y_p + 1/v` reduction runs into.
        Err(DsolveError::QuadratureFailed(m)) => {
            assert!(
                m.contains("riccati"),
                "decline does not name the class: {m}"
            );
            assert!(
                m.contains("exp(x^2)"),
                "decline does not name the integral: {m}"
            );
        }
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
    solution_is_verified(&input, &sol, &pool)
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
            res.solutions[0].render(&pool)
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
    // original equation independently of that gate.
    //
    // Which half of the gate certifies them is asserted, not assumed.  It is
    // tempting to reason that `verify::eval_func` has no `f64` kernel for
    // `Ei`/`Si`/`Ci`, so the numeric fallback cannot fire and these must pass
    // on an exact symbolic `residual ≡ 0`.  That is wrong, and measurably so:
    // the special functions cancel *out of the residual* when the candidate is
    // substituted, leaving an elementary expression the sampler evaluates
    // perfectly well — `x⁻¹·eˣ·e⁻ˣ − x⁻¹` for the first — which is not the
    // symbolic zero, because nothing in the default rule set collapses
    // `eˣ·e⁻ˣ`.  **Both** of these are certified by the numeric fallback, which
    // is load-bearing rather than unreachable; the same is true of
    // `y'' − 4y = 1/x` and `y''' − y' = 1/x` in the corpus.
    for (src, basis) in [
        ("ypp - y - 1/x", &["Ei"] as &[&str]),
        // `basis_functions_used` returns the names sorted.
        ("ypp + y - 1/x", &["Ci", "Si"]),
    ] {
        let (pool, input, sol) = solve_src(2, src);
        assert_eq!(sol.constants.len(), 2, "`{src}`: wrong constant count");
        assert!(
            !super::verify::certifies_symbolically(&input, explicit_of(&sol), &pool),
            "`{src}`: now certified symbolically — the comment above is stale",
        );
        assert_eq!(
            basis_functions_used(explicit_of(&sol), &pool),
            basis,
            "`{src}`: wrong special-function vocabulary in {}",
            pool.display(explicit_of(&sol))
        );
        for c in &sol.constants {
            assert!(
                super::contains(explicit_of(&sol), *c, &pool),
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

// ---------------------------------------------------------------------------
// First-order classes: the standard cascade
// ---------------------------------------------------------------------------

/// Solve `src` and return `(pool, input, solution)` without asserting the form,
/// re-verifying independently of the gate inside `dsolve`.
fn solve1(src: &str) -> (ExprPool, OdeInput, DsolveSolution) {
    solve_src(1, src)
}

#[test]
fn michaelis_menten_elimination() {
    // `(Kₘ + y)·y' + Vₘ·y = 0` — saturable elimination, the workhorse of
    // pharmacokinetics, and until now `no implemented first-order class
    // matched`.  It separates to `Kₘ·log y + y = C − Vₘ·x`, which the Lambert
    // inversion turns into `y = Kₘ·W(e^{(C − Vₘx)/Kₘ}/Kₘ)` — SymPy's answer.
    //
    // Both spellings are checked: the equation as a chemist writes it, and the
    // `y' = −Vₘy/(Kₘ+y)` form a modeller types.
    for src in ["(Km + y)*yp + Vm*y", "yp + Vm*y/(Km + y)"] {
        let (pool, _, sol) = solve1(src);
        assert_eq!(sol.method, "separable", "`{src}`");
        let y_of_x = explicit_of(&sol);
        assert!(
            pool.display(y_of_x).to_string().contains("lambert_w"),
            "`{src}`: expected a Lambert-W form, got {}",
            pool.display(y_of_x)
        );
        assert_eq!(sol.constants.len(), 1, "`{src}`");
        assert!(super::contains(y_of_x, sol.constants[0], &pool), "`{src}`");
    }
}

#[test]
fn logistic_growth_with_symbolic_parameters() {
    // `y' = r·y·(1 − y/K)`.  The textbook `r = K = 1` case solved before; with
    // symbolic parameters it did not, for two independent reasons that both had
    // to go: the `∫ dy/h(y)` inversion insisted on literal `±1` log
    // coefficients, and the verification gate had nothing to bind `r` and `K`
    // to, so the numeric half could not run at all.
    for src in ["yp - r*y*(1 - y/K)", "yp - k*y*(1 - y/K)"] {
        let (pool, _, sol) = solve1(src);
        assert_eq!(sol.constants.len(), 1, "`{src}`");
        let y_of_x = explicit_of(&sol);
        for name in ["K", "exp"] {
            assert!(
                pool.display(y_of_x).to_string().contains(name),
                "`{src}`: expected `{name}` in {}",
                pool.display(y_of_x)
            );
        }
    }
    // The numeric-parameter logistic still solves, explicitly.
    let (_, _, sol) = solve1("yp - y*(1 - y)");
    assert!(sol.is_explicit());
}

#[test]
fn linear_handles_symbolic_and_variable_coefficients() {
    // `p` and `q` constant but symbolic, then genuinely non-constant.
    for (src, want) in [
        ("yp + p*y - q", "p"),
        ("yp + ke*y", "ke"),
        ("yp + y/x - x^2", "x"),
        ("yp - y/x - x*log(x)", "log"),
    ] {
        let (pool, _, sol) = solve1(src);
        let y_of_x = explicit_of(&sol);
        assert!(
            pool.display(y_of_x).to_string().contains(want),
            "`{src}`: expected `{want}` in {}",
            pool.display(y_of_x)
        );
        assert_eq!(sol.constants.len(), 1, "`{src}`");
    }
}

#[test]
fn bernoulli_and_riccati_still_close() {
    for src in ["yp + y - y^2", "yp - y - x*y^2", "yp - exp(x)*y^2"] {
        let (_, _, sol) = solve1(src);
        assert!(sol.is_explicit(), "`{src}`");
    }
    // Riccati with `y_p = x`.
    let (_, _, sol) = solve1("yp - y^2 + 2*x*y - x^2 - 1");
    assert_eq!(sol.method, "riccati");
}

#[test]
fn exact_integrating_factor_rescues() {
    // Near-exact equations: `M dx + N dy` is fixed only up to a common factor
    // by `y' = f(x, y)`, and the arbitrary choice is almost never the exact
    // one.  Each of these needs one of the two standard rescues.
    for (src, method) in [
        // μ(y) = y⁻⁴
        ("(2*x*y) + (y^2 - 3*x^2)*yp", "exact_integrating_factor_y"),
        // μ(x) = x
        ("(x^2 + y^2 + x) + x*y*yp", "exact_integrating_factor_x"),
        // μ(y) = y
        ("y + (2*x - y*exp(y))*yp", "exact_integrating_factor_y"),
    ] {
        let (_, _, sol) = solve1(src);
        assert_eq!(sol.method, method, "`{src}`");
        assert_eq!(sol.constants.len(), 1, "`{src}`");
    }
    // And a genuinely exact one still goes through the plain route.
    let (_, _, sol) = solve1("(2*x + y) + (x + 2*y)*yp");
    assert_eq!(sol.method, "exact");
}

#[test]
fn homogeneous_quotients_are_recognised() {
    // `(x+y)/(x−y)` is homogeneous of degree zero, but substituting `y = v·x`
    // leaves an `x` that no rule set cancels — the reciprocal wraps a *sum*.
    // The class used to decline two equations out of its own chapter.
    for src in [
        "yp - (x^2 + y^2)/(x*y)",
        "yp - (x + y)/(x - y)",
        "yp - (x + 3*y)/(x - y)",
    ] {
        let (_, _, sol) = solve1(src);
        assert_eq!(sol.method, "homogeneous", "`{src}`");
    }
    // The same equation with the quotient cleared is solved too — by the exact
    // class, which comes first and also handles it.  What matters is that it
    // is answered and verified, not which of the two overlapping classes wins.
    let (_, _, sol) = solve1("yp*y*x - y^2 - x^2");
    assert!(sol.is_explicit());
}

#[test]
fn implicit_solutions_are_labelled_rather_than_disguised() {
    // Separable and exact equations often have no closed form for `y`.  The
    // answer is then the relation, and the API must make that impossible to
    // misread: `y_of_x()` is `None` and the relation is somewhere else.
    for src in [
        // ∫ dy/sin y = log tan(y/2): no inverse this code can write down.
        "yp - sin(x)*sin(y)",
        // exact, potential eˣ·sin y = C.
        "exp(x)*sin(y) + exp(x)*cos(y)*yp",
        // near-exact, rescued by μ(y) = y; the potential mixes `y²eʸ` with
        // `x·y²` and is not solvable for `y` by anything here.
        "y + (2*x - y*exp(y))*yp",
    ] {
        let (pool, input, sol) = solve1(src);
        assert!(!sol.is_explicit(), "`{src}`: expected an implicit answer");
        assert!(sol.y_of_x().is_none(), "`{src}`: y_of_x must be None");
        let rel = sol
            .implicit_relation()
            .expect("an implicit solution carries a relation");
        assert!(super::contains(rel, input.y, &pool), "`{src}`");
        assert!(super::contains(rel, sol.constants[0], &pool), "`{src}`");
        // The relation verifies as a relation, through the implicit gate.
        implicit_relation_is_zero(&input, rel, &sol.constants, &pool)
            .unwrap_or_else(|e| panic!("`{src}`: relation does not verify: {e}"));
    }
}

#[test]
fn an_explicit_answer_wins_over_an_implicit_one() {
    // `y' = eˣ·y²` separates to `−1/y = eˣ + C`, which this code does not
    // invert and would hand back as a relation; Bernoulli solves it for `y`.
    // The cascade must not stop at the first class that merely *answers*.
    let (pool, _, sol) = solve1("yp - exp(x)*y^2");
    assert!(sol.is_explicit());
    assert_eq!(sol.method, "bernoulli");
    // …and the constant is `C1`, not `C2`: a class that tried and did not
    // answer must not leave its footprint in the answer.
    assert_eq!(sol.constants.len(), 1);
    assert_eq!(pool.display(sol.constants[0]).to_string(), "C1");
}

/// Build `y' = y` and a relation in `x`, `y`, `C1`, for the implicit-gate
/// tests.  Returns `(pool, input, relation, c1)`.
fn implicit_fixture(relation_src: &str) -> (ExprPool, OdeInput, ExprId, ExprId) {
    use crate::parse::parse;
    use std::collections::HashMap;
    let pool = ExprPool::new();
    let input = super::corpus::build_ode(1, "yp - y", &pool).expect("parses");
    let c1 = pool.symbol("C1", Domain::Real);
    let mut syms: HashMap<String, ExprId> = HashMap::new();
    syms.insert("x".to_owned(), input.x);
    syms.insert("y".to_owned(), input.y);
    syms.insert("C1".to_owned(), c1);
    let rel = parse(relation_src, &pool, &mut syms).expect("relation parses");
    let rel = simp(rel, &pool);
    (pool, input, rel, c1)
}

#[test]
fn the_implicit_gate_certifies_a_relation_and_refuses_a_wrong_one() {
    // `y' = y` has the general solution `log y − x = C`.
    let (pool, input, rel, c1) = implicit_fixture("log(y) - x - C1");
    implicit_relation_is_zero(&input, rel, &[c1], &pool).expect("the true relation must certify");

    // Every one of these is wrong for `y' = y`, and each is wrong in a way the
    // gate has to catch on its own terms: a rescaled slope field, a relation
    // belonging to a different equation, and one that is not a family at all.
    for src in [
        "log(y) - 2*x - C1", // slope 2y, not y
        "y^2 + x^2 - C1",    // the circle: y' = −x/y
        "log(y) - x",        // no constant — a single curve, not a general solution
        "x - C1",            // no y at all
    ] {
        let (pool, input, rel, c1) = implicit_fixture(src);
        assert!(
            implicit_relation_is_zero(&input, rel, &[c1], &pool).is_err(),
            "`{src}` must not certify as a general solution of y' = y"
        );
    }
}

#[test]
fn the_implicit_gate_refuses_a_slope_field_that_mentions_the_constant() {
    // `y − C·eˣ = 0` *is* a general solution of `y' = y`, and the gate refuses
    // it anyway: `−Gₓ/G_y = C·eˣ` is only the right slope *on* that curve, so
    // the free-`(x, y)` sampling the gate is built on cannot be run.  Refusing
    // is the conservative direction — no wrong answer is returned, the class
    // simply falls through to a form that can be checked (here `y = C·eˣ`,
    // which the explicit gate takes).  Pinned so the precondition is not
    // quietly dropped later.
    let (pool, input, rel, c1) = implicit_fixture("y - C1*exp(x)");
    let err = implicit_relation_is_zero(&input, rel, &[c1], &pool)
        .expect_err("a constant-dependent slope field must be refused");
    assert!(
        format!("{err}").contains("integration constant"),
        "refusal does not say why: {err}"
    );
}

#[test]
fn riccati_without_a_particular_solution_refuses_by_name() {
    // `y' = y² + x` is the Airy equation in disguise.  Refusing is correct;
    // refusing with "no implemented first-order class matched" is not, because
    // the class *was* recognised and the missing piece is nameable.
    let pool = ExprPool::new();
    let input = super::corpus::build_ode(1, "yp - y^2 - x", &pool).expect("parses");
    match dsolve(&input, &pool) {
        Err(DsolveError::NoParticularSolution(m)) => {
            assert!(
                m.contains("riccati"),
                "decline does not name the class: {m}"
            );
            assert!(
                m.contains("particular solution"),
                "decline does not name what is missing: {m}"
            );
        }
        Err(e) => panic!("expected a named Riccati refusal, got {e}"),
        Ok(res) => panic!("expected a refusal, got {}", res.solutions[0].render(&pool)),
    }
}

#[test]
fn a_failed_inner_integral_is_reported_as_such() {
    // The decline has to say *which* integral did not close.  `y' = 2xy + …`
    // needs `∫ e^{x²}` through the Riccati reduction; the class is named and
    // the integrand quoted, so the report points at the integration engine
    // rather than at the classifier.
    //
    // Premise: `∫ e^{x²} dx` is not elementary and Alkahest emits no `erfi`.
    // If that changes the equation will *solve*, and this test should be
    // rewritten around whatever the integrator then declines — the decline
    // path is what is worth keeping, not this particular integrand.
    let pool = ExprPool::new();
    let input = super::corpus::build_ode(1, "yp - y^2 + x^2 - 1", &pool).expect("parses");
    match dsolve(&input, &pool) {
        Err(DsolveError::QuadratureFailed(m)) => {
            assert!(m.contains('∫'), "decline does not quote an integral: {m}");
        }
        Err(e) => panic!("expected a quadrature decline, got {e}"),
        Ok(_) => {}
    }
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
                super::contains(explicit_of(&sol), *c, &pool),
                "`{src}`: constant {} does not appear in the solution",
                pool.display(*c)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Constant coefficients that are *symbolic*
// ---------------------------------------------------------------------------

/// Solve `src = 0` through [`dsolve_with`] and return the report, asserting the
/// first branch verifies independently.
fn report_src(order: usize, src: &str) -> (ExprPool, OdeInput, DsolveReport) {
    let pool = ExprPool::new();
    let input = super::corpus::build_ode(order, src, &pool).expect("source parses");
    let rep = dsolve_with(&input, &AssumptionContext::new(), &pool)
        .unwrap_or_else(|e| panic!("`{src}` should solve: {e}"));
    let sol = &rep.result.solutions[0];
    residual_is_zero(
        &input,
        sol.y_of_x().expect("an explicit solution"),
        &sol.constants,
        &pool,
    )
    .unwrap_or_else(|e| panic!("`{src}` returned an unverified solution: {e}"));
    (pool, input, rep)
}

fn nonzero_conditions(rep: &DsolveReport, pool: &ExprPool) -> Vec<String> {
    rep.side_conditions
        .iter()
        .map(|c| match c {
            SideCondition::NonZero(id) => pool.display(*id).to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

/// The damped harmonic oscillator — the case 3.9.0 refused outright.
#[test]
fn damped_oscillator_with_symbolic_zeta_and_omega() {
    let (pool, _, rep) = report_src(2, "ypp + 2*z*w*yp + w^2*y");
    let sol = &rep.result.solutions[0];
    assert_eq!(sol.constants.len(), 2);
    assert_eq!(sol.method, "constant_coefficient_symbolic");
    // Both exponentials must be present and must involve both parameters.
    let y = pool
        .display(sol.y_of_x().expect("an explicit solution"))
        .to_string();
    assert_eq!(
        y.matches("exp(").count(),
        2,
        "expected two exponentials: {y}"
    );
    // The critically-damped branch is stated, not assumed away.
    assert_eq!(
        rep.side_conditions.len(),
        1,
        "expected exactly the discriminant condition, got {:?}",
        nonzero_conditions(&rep, &pool)
    );
    assert!(matches!(rep.side_conditions[0], SideCondition::NonZero(_)));
    assert!(
        rep.notes.iter().any(|n| n.contains("discriminant")),
        "the repeated-root case must be named in prose, got {:?}",
        rep.notes
    );
}

/// A discriminant that is *provably* zero takes the confluent branch and
/// carries no condition at all.
#[test]
fn a_provable_double_root_gets_the_secular_solution() {
    let (pool, _, rep) = report_src(2, "ypp + 2*a*yp + a^2*y");
    let sol = &rep.result.solutions[0];
    assert_eq!(sol.method, "constant_coefficient_symbolic_repeated_root");
    assert!(
        rep.side_conditions.is_empty(),
        "nothing is undecided here: {:?}",
        nonzero_conditions(&rep, &pool)
    );
    // The second basis function must carry the secular factor `x`.
    let y = pool
        .display(sol.y_of_x().expect("an explicit solution"))
        .to_string();
    assert!(
        y.contains("x * exp") || y.contains("exp(") && y.contains("x *"),
        "expected an x·e^{{−ax}} term, got {y}"
    );
}

/// A symbolic *leading* coefficient is an assumption about the equation's
/// order, and is reported as one.
#[test]
fn a_symbolic_leading_coefficient_is_a_stated_assumption() {
    let (pool, _, rep) = report_src(2, "a*ypp + b*yp + c*y");
    let conds = nonzero_conditions(&rep, &pool);
    assert!(
        conds.iter().any(|c| c == "a"),
        "`a ≠ 0` must be stated — at a = 0 the equation is first order, not \
         second — got {conds:?}"
    );
    assert_eq!(
        conds.len(),
        2,
        "leading coefficient and discriminant: {conds:?}"
    );
}

/// A caller who rules out the confluence gets an unconditional answer.
#[test]
fn an_asserted_nonzero_discriminant_removes_the_condition() {
    let pool = ExprPool::new();
    let input = super::corpus::build_ode(2, "ypp + 2*z*w*yp + w^2*y", &pool).expect("parses");
    // The solver forms D = 4ζ²ω² − 4ω²; the caller states 4ω²(ζ²−1) ≠ 0 in the
    // equivalent spelling ζ²ω² − ω² ≠ 0, which differs by the factor 4.
    let z = pool.symbol("z", Domain::Real);
    let w = pool.symbol("w", Domain::Real);
    let w2 = pool.pow(w, pool.integer(2_i32));
    let d = pool.add(vec![
        pool.mul(vec![pool.pow(z, pool.integer(2_i32)), w2]),
        pool.mul(vec![pool.integer(-1_i32), w2]),
    ]);
    let mut assumptions = AssumptionContext::new();
    assumptions
        .refine(pool.pred_ne(d, pool.integer(0_i32)), &pool)
        .expect("satisfiable");
    let rep = dsolve_with(&input, &assumptions, &pool).expect("solves");
    assert!(
        rep.side_conditions.is_empty(),
        "the caller settled the branch, got {:?}",
        nonzero_conditions(&rep, &pool)
    );
    assert_eq!(
        rep.result.solutions[0].method,
        "constant_coefficient_symbolic_distinct_roots"
    );
}

/// A stated *negative* discriminant selects the real oscillatory form rather
/// than complex exponentials.
#[test]
fn an_asserted_negative_discriminant_gives_the_real_oscillatory_form() {
    let pool = ExprPool::new();
    let input = super::corpus::build_ode(2, "ypp + 2*z*w*yp + w^2*y", &pool).expect("parses");
    let z = pool.symbol("z", Domain::Real);
    let w = pool.symbol("w", Domain::Real);
    let w2 = pool.pow(w, pool.integer(2_i32));
    // −D/4 = ω² − ζ²ω² > 0, i.e. the underdamped region.
    let neg_d = pool.add(vec![
        w2,
        pool.mul(vec![
            pool.integer(-1_i32),
            pool.pow(z, pool.integer(2_i32)),
            w2,
        ]),
    ]);
    let mut assumptions = AssumptionContext::new();
    assumptions
        .refine(pool.pred_gt(neg_d, pool.integer(0_i32)), &pool)
        .expect("satisfiable");
    let rep = dsolve_with(&input, &assumptions, &pool).expect("solves");
    let sol = &rep.result.solutions[0];
    assert_eq!(sol.method, "constant_coefficient_symbolic_oscillatory");
    let y = pool
        .display(sol.y_of_x().expect("an explicit solution"))
        .to_string();
    assert!(y.contains("cos(") && y.contains("sin("), "got {y}");
    residual_is_zero(
        &input,
        sol.y_of_x().expect("an explicit solution"),
        &sol.constants,
        &pool,
    )
    .expect("the oscillatory form must verify too");
    assert!(rep.side_conditions.is_empty());
}

/// The `λᵏ` factor is peeled off, so a symbolic third-order equation with no
/// `y` term still reduces to a quadratic.
#[test]
fn a_lambda_factor_reduces_the_symbolic_degree() {
    let (pool, _, rep) = report_src(3, "yppp + 2*z*w*ypp + w^2*yp");
    let sol = &rep.result.solutions[0];
    assert_eq!(sol.constants.len(), 3);
    let y = pool
        .display(sol.y_of_x().expect("an explicit solution"))
        .to_string();
    assert_eq!(
        y.matches("exp(").count(),
        2,
        "one constant mode + two exponentials: {y}"
    );
}

/// Beyond degree two there is no closed form, and none is invented.
#[test]
fn a_symbolic_cubic_characteristic_polynomial_is_refused() {
    let pool = ExprPool::new();
    let input = super::corpus::build_ode(3, "yppp + a*ypp + b*yp + c*y", &pool).expect("parses");
    match dsolve(&input, &pool) {
        Err(DsolveError::Unsupported(m)) => {
            assert!(
                m.contains("degree"),
                "the message must name the degree: {m}"
            );
        }
        Err(e) => panic!("expected an Unsupported decline, got {e}"),
        Ok(r) => panic!(
            "expected a decline, got {}",
            pool.display(r.solutions[0].y_of_x().expect("an explicit solution"))
        ),
    }
}

/// A numeric equation is untouched by the symbolic route: same method label,
/// same real cos/sin output, no conditions.
#[test]
fn numeric_coefficients_still_take_the_rational_route() {
    let (pool, _, rep) = report_src(2, "ypp + 2*yp + 5*y");
    let sol = &rep.result.solutions[0];
    assert_eq!(sol.method, "constant_coefficient");
    assert!(rep.side_conditions.is_empty());
    let y = pool
        .display(sol.y_of_x().expect("an explicit solution"))
        .to_string();
    assert!(y.contains("cos(") && y.contains("sin("), "got {y}");
}

/// Every symbolic corpus entry solves *and* verifies independently.
#[test]
fn symbolic_corpus_entries_all_verify() {
    for (class, name, order, src) in super::corpus::CORPUS {
        if *class != "cc-sym" {
            continue;
        }
        let pool = ExprPool::new();
        let input = super::corpus::build_ode(*order, src, &pool).expect("parses");
        let res = dsolve(&input, &pool).unwrap_or_else(|e| panic!("`{name}` should solve: {e}"));
        let sol = &res.solutions[0];
        assert_eq!(sol.constants.len(), *order, "`{name}` constant count");
        residual_is_zero(
            &input,
            sol.y_of_x().expect("an explicit solution"),
            &sol.constants,
            &pool,
        )
        .unwrap_or_else(|e| panic!("`{name}` returned an unverified solution: {e}"));
    }
}

/// The parametric gate must **refuse** a candidate that is wrong in the
/// parameters even though it is right at one convenient value of them.
#[test]
fn a_candidate_right_at_one_parameter_value_only_is_refused() {
    let pool = ExprPool::new();
    let input = super::corpus::build_ode(2, "ypp + 2*z*w*yp + w^2*y", &pool).expect("parses");
    // y = C1·e^{−ωx}: a solution exactly when ζ = 1, not in general.
    let w = pool.symbol("w", Domain::Real);
    let c1 = pool.symbol("C1", Domain::Real);
    let arg = pool.mul(vec![pool.integer(-1_i32), w, input.x]);
    let bogus = pool.mul(vec![c1, pool.func("exp", vec![arg])]);
    assert!(
        residual_is_zero(&input, bogus, &[c1], &pool).is_err(),
        "a candidate that only works at ζ = 1 must not be certified"
    );
}
