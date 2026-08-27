//! Tests for the Risch–Norman heuristic.
//!
//! Three groups, matching the three ways the module can be wrong:
//!
//! 1. **Cross-check** — everything the existing engine solves must be solved
//!    or cleanly declined here, never answered *differently*.  Comparison is
//!    by differentiation, never by string matching.
//! 2. **New coverage** — integrands the current engine declines.
//! 3. **Negative** — genuinely non-elementary integrands must decline, and the
//!    decline must not be convertible into a certificate.

use super::*;
use crate::integrate::{integrate, verify_antiderivative_exact};
use crate::kernel::{Domain, ExprPool};

fn setup() -> (ExprPool, ExprId) {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    (pool, x)
}

fn exp_of(pool: &ExprPool, a: ExprId) -> ExprId {
    pool.func("exp", vec![a])
}

fn log_of(pool: &ExprPool, a: ExprId) -> ExprId {
    pool.func("log", vec![a])
}

fn inv(pool: &ExprPool, a: ExprId) -> ExprId {
    pool.pow(a, pool.integer(-1_i32))
}

/// Assert the heuristic solves `f` and that the answer really differentiates
/// back to `f`.
fn assert_solves(pool: &ExprPool, f: ExprId, x: ExprId) -> ExprId {
    match integrate_parallel_risch(f, x, pool) {
        ParallelRischOutcome::Solved { antiderivative, .. } => {
            assert!(
                verify_antiderivative_status(antiderivative, f, x, pool).is_some(),
                "returned antiderivative does not differentiate back to the integrand"
            );
            antiderivative
        }
        ParallelRischOutcome::Declined(r) => panic!("expected a solution, got decline: {r}"),
    }
}

/// Assert the heuristic solves `f` **and** that the identity was established
/// exactly (the ring-level gate), not just by numeric sampling.
fn assert_solves_exactly(pool: &ExprPool, f: ExprId, x: ExprId) -> ExprId {
    match integrate_parallel_risch(f, x, pool) {
        ParallelRischOutcome::Solved {
            antiderivative,
            verification,
        } => {
            assert_eq!(
                verification,
                AntiderivativeVerification::Exact,
                "expected an exact identity, got {verification:?} for {}",
                crate::kernel::display::render_unicode(antiderivative, pool)
            );
            antiderivative
        }
        ParallelRischOutcome::Declined(r) => panic!("expected a solution, got decline: {r}"),
    }
}

fn assert_declines(pool: &ExprPool, f: ExprId, x: ExprId) -> DeclineReason {
    match integrate_parallel_risch(f, x, pool) {
        ParallelRischOutcome::Solved { antiderivative, .. } => panic!(
            "expected a decline, got {}",
            crate::kernel::display::render_unicode(antiderivative, pool)
        ),
        ParallelRischOutcome::Declined(r) => r,
    }
}

// ---------------------------------------------------------------------------
// 1. Cross-check against the existing engine
// ---------------------------------------------------------------------------

#[test]
fn polynomial_matches_engine() {
    let (pool, x) = setup();
    // ∫ 3x² + 2x + 1
    let f = pool.add(vec![
        pool.mul(vec![pool.integer(3_i32), pool.pow(x, pool.integer(2_i32))]),
        pool.mul(vec![pool.integer(2_i32), x]),
        pool.integer(1_i32),
    ]);
    let got = assert_solves(&pool, f, x);
    // `d/dx F` must be *structurally* the integrand — the strongest form of the
    // check available, and stronger than `verify_antiderivative_exact`, whose
    // `simplify(d − f) == 0` residual test does not close on this input.
    let d = crate::diff::diff(got, x, &pool)
        .expect("differentiable")
        .value;
    assert_eq!(crate::simplify::engine::simplify(d, &pool).value, f);
    // The engine solves it too; both are antiderivatives of the same integrand.
    let engine = integrate(f, x, &pool)
        .expect("engine solves polynomials")
        .value;
    let de = crate::diff::diff(engine, x, &pool)
        .expect("differentiable")
        .value;
    assert_eq!(crate::simplify::engine::simplify(de, &pool).value, f);
}

#[test]
fn one_over_x_matches_engine() {
    let (pool, x) = setup();
    let f = inv(&pool, x);
    let got = assert_solves(&pool, f, x);
    assert!(verify_antiderivative_exact(got, f, x, &pool));
}

#[test]
fn x_log_x_matches_engine() {
    let (pool, x) = setup();
    let f = pool.mul(vec![x, log_of(&pool, x)]);
    let got = assert_solves(&pool, f, x);
    assert!(verify_antiderivative_status(got, f, x, &pool).is_some());
    assert!(integrate(f, x, &pool).is_ok(), "engine also solves this");
}

#[test]
fn poly_times_exp_matches_engine() {
    let (pool, x) = setup();
    // ∫ x²·exp(x)
    let f = pool.mul(vec![pool.pow(x, pool.integer(2_i32)), exp_of(&pool, x)]);
    let got = assert_solves(&pool, f, x);
    assert!(verify_antiderivative_status(got, f, x, &pool).is_some());
}

#[test]
fn gaussian_times_cubic_matches_engine() {
    let (pool, x) = setup();
    // ∫ x³·exp(−x²) — the engine solves this; so must we.
    let negx2 = pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))]);
    let f = pool.mul(vec![pool.pow(x, pool.integer(3_i32)), exp_of(&pool, negx2)]);
    let got = assert_solves(&pool, f, x);
    assert!(verify_antiderivative_status(got, f, x, &pool).is_some());
}

#[test]
fn rational_with_repeated_pole() {
    let (pool, x) = setup();
    // ∫ 1/(x+1)²
    let xp1 = pool.add(vec![x, pool.integer(1_i32)]);
    let f = pool.pow(xp1, pool.integer(-2_i32));
    let got = assert_solves(&pool, f, x);
    assert!(verify_antiderivative_exact(got, f, x, &pool));
}

// ---------------------------------------------------------------------------
// 2. New coverage: cases the current engine declines
// ---------------------------------------------------------------------------

#[test]
fn exp_over_exp_plus_one() {
    let (pool, x) = setup();
    // ∫ exp(x)/(exp(x)+1) = log(exp(x)+1) — `E-INT-001` from the engine.
    let e = exp_of(&pool, x);
    let den = pool.add(vec![e, pool.integer(1_i32)]);
    let f = pool.mul(vec![e, inv(&pool, den)]);
    let got = assert_solves(&pool, f, x);
    assert!(verify_antiderivative_status(got, f, x, &pool).is_some());
}

#[test]
fn exp_two_x_over_exp_plus_one() {
    let (pool, x) = setup();
    // ∫ exp(2x)/(exp(x)+1) = exp(x) − log(exp(x)+1)
    let two_x = pool.mul(vec![pool.integer(2_i32), x]);
    let e2 = exp_of(&pool, two_x);
    let e = exp_of(&pool, x);
    let den = pool.add(vec![e, pool.integer(1_i32)]);
    let f = pool.mul(vec![e2, inv(&pool, den)]);
    let got = assert_solves(&pool, f, x);
    assert!(verify_antiderivative_status(got, f, x, &pool).is_some());
}

#[test]
fn one_over_one_plus_exp_minus_x() {
    let (pool, x) = setup();
    // ∫ 1/(1 + exp(−x)) = log(1 + exp(x)); exercises the exponential lattice
    // (exp(−x) and exp(x) must become a single generator).
    let negx = pool.mul(vec![pool.integer(-1_i32), x]);
    let em = exp_of(&pool, negx);
    let den = pool.add(vec![pool.integer(1_i32), em]);
    let f = inv(&pool, den);
    let got = assert_solves(&pool, f, x);
    assert!(verify_antiderivative_status(got, f, x, &pool).is_some());
}

#[test]
fn one_over_x_log_x_written_as_a_product_inverse() {
    let (pool, x) = setup();
    // ∫ 1/(x·log x) = log(log x); `(x*log(x))^(-1)` is the spelling the
    // engine's routing declines.
    let l = log_of(&pool, x);
    let f = inv(&pool, pool.mul(vec![x, l]));
    let got = assert_solves(&pool, f, x);
    assert!(verify_antiderivative_status(got, f, x, &pool).is_some());
}

// ---------------------------------------------------------------------------
// 3. Negative tests — a decline must never become a certificate
// ---------------------------------------------------------------------------

#[test]
fn gaussian_declines_and_is_not_a_certificate() {
    let (pool, x) = setup();
    // ∫ exp(x²) — genuinely non-elementary.
    let f = exp_of(&pool, pool.pow(x, pool.integer(2_i32)));
    let reason = assert_declines(&pool, f, x);
    // The only error this can become is `NotImplemented`.
    let err = reason.into_integration_error();
    assert!(
        matches!(err, IntegrationError::NotImplemented(_)),
        "a decline must never become a NonElementary certificate, got {err:?}"
    );
}

#[test]
fn exp_over_x_declines_and_is_not_a_certificate() {
    let (pool, x) = setup();
    // ∫ exp(x)/x — the exponential integral Ei; non-elementary.
    let f = pool.mul(vec![exp_of(&pool, x), inv(&pool, x)]);
    let reason = assert_declines(&pool, f, x);
    assert!(matches!(
        reason.clone().into_integration_error(),
        IntegrationError::NotImplemented(_)
    ));
}

#[test]
fn x_over_exp_plus_one_declines() {
    let (pool, x) = setup();
    // ∫ x/(exp(x)+1) — polylogarithmic, non-elementary.
    let e = exp_of(&pool, x);
    let den = pool.add(vec![e, pool.integer(1_i32)]);
    let f = pool.mul(vec![x, inv(&pool, den)]);
    let reason = assert_declines(&pool, f, x);
    assert!(matches!(
        reason.into_integration_error(),
        IntegrationError::NotImplemented(_)
    ));
}

#[test]
fn log_x_over_one_plus_x_declines() {
    let (pool, x) = setup();
    // ∫ log(x)/(1+x) = log x·log(1+x) + Li₂(−x) — non-elementary.
    let l = log_of(&pool, x);
    let den = pool.add(vec![pool.integer(1_i32), x]);
    let f = pool.mul(vec![l, inv(&pool, den)]);
    assert_declines(&pool, f, x);
}

#[test]
fn log_log_x_declines() {
    let (pool, x) = setup();
    // ∫ log(log x) = x·log(log x) − li(x) — non-elementary.
    let f = log_of(&pool, log_of(&pool, x));
    assert_declines(&pool, f, x);
}

#[test]
fn no_outcome_variant_can_express_non_elementarity() {
    // A compile-time-ish statement of the contract: exhaustively matching the
    // outcome yields only `Solved` and `Declined`, and every `DeclineReason`
    // maps to `NotImplemented`.
    for reason in [
        DeclineReason::UnsupportedIntegrand("t".to_string()),
        DeclineReason::DependentGenerators("t"),
        DeclineReason::NoSolution,
        DeclineReason::LinearSolver,
        DeclineReason::TooLarge("t"),
        DeclineReason::RingArithmetic,
        DeclineReason::VerificationFailed,
    ] {
        assert!(matches!(
            reason.into_integration_error(),
            IntegrationError::NotImplemented(_)
        ));
    }
}

// ---------------------------------------------------------------------------
// 4. Out-of-ring and degenerate cases
// ---------------------------------------------------------------------------

#[test]
fn sqrt_tan_declines_cleanly() {
    let (pool, x) = setup();
    // √(tan x): outside the ring on two counts (trig generator, half power).
    let t = pool.func("tan", vec![x]);
    let f = pool.pow(t, pool.rational(1_i32, 2_i32));
    let reason = assert_declines(&pool, f, x);
    assert!(matches!(reason, DeclineReason::UnsupportedIntegrand(_)));
}

#[test]
fn sin_declines_as_out_of_ring() {
    let (pool, x) = setup();
    let f = pool.func("sin", vec![x]);
    let reason = assert_declines(&pool, f, x);
    assert!(matches!(reason, DeclineReason::UnsupportedIntegrand(_)));
}

#[test]
fn dependent_logarithms_decline() {
    let (pool, x) = setup();
    // log(2x) and log(x) are multiplicatively dependent modulo constants, so
    // the monomial set is degenerate and coefficient matching is not valid.
    let two_x = pool.mul(vec![pool.integer(2_i32), x]);
    let l2 = log_of(&pool, two_x);
    let l1 = log_of(&pool, x);
    let f = pool.mul(vec![l2, l1]);
    let reason = assert_declines(&pool, f, x);
    assert!(
        matches!(reason, DeclineReason::DependentGenerators(_)),
        "expected a dependence decline, got {reason}"
    );
}

#[test]
fn exp_of_a_logarithm_is_detected_as_dependent() {
    let (pool, x) = setup();
    // exp(2·log x) = x² — the exponential is algebraic over the field, so the
    // structure-theorem check must fire rather than the linear system.
    let l = log_of(&pool, x);
    let arg = pool.mul(vec![pool.integer(2_i32), l]);
    let e = exp_of(&pool, arg);
    let f = pool.mul(vec![e, l]);
    let reason = assert_declines(&pool, f, x);
    assert!(
        matches!(reason, DeclineReason::DependentGenerators(_)),
        "expected a dependence decline, got {reason}"
    );
}

#[test]
fn arctan_case_declines_because_the_constant_field_is_q() {
    let (pool, x) = setup();
    // ∫ 1/(x²+1) = arctan x — elementary, but not reachable over ℚ.  This is
    // the canonical reminder that a decline is not a verdict.
    let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
    let f = inv(&pool, den);
    let reason = assert_declines(&pool, f, x);
    assert!(matches!(reason, DeclineReason::NoSolution));
    // …and the engine *does* solve it, which is the point.
    assert!(integrate(f, x, &pool).is_ok());
}

// ---------------------------------------------------------------------------
// 5. The exact (ring-level) gate
// ---------------------------------------------------------------------------

#[test]
fn answers_are_gated_by_an_exact_ring_identity() {
    // These all normalise cleanly back into `ℚ(x, θ)`, so the ring-level gate
    // — not the numeric fallback — is what accepts them.  Pinning this keeps
    // the module from quietly regressing onto sampled agreement.
    let (pool, x) = setup();
    let cases: Vec<ExprId> = vec![
        // ∫ x²
        pool.pow(x, pool.integer(2_i32)),
        // ∫ 1/(x+1)²
        {
            let xp1 = pool.add(vec![x, pool.integer(1_i32)]);
            pool.pow(xp1, pool.integer(-2_i32))
        },
        // ∫ x³·exp(−x²)
        {
            let negx2 = pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))]);
            pool.mul(vec![pool.pow(x, pool.integer(3_i32)), exp_of(&pool, negx2)])
        },
        // ∫ exp(2x)/(exp(x)+1)
        {
            let e = exp_of(&pool, x);
            let e2 = exp_of(&pool, pool.mul(vec![pool.integer(2_i32), x]));
            let den = pool.add(vec![e, pool.integer(1_i32)]);
            pool.mul(vec![e2, inv(&pool, den)])
        },
        // ∫ 1/(x·log x)
        inv(&pool, pool.mul(vec![x, log_of(&pool, x)])),
    ];
    for f in cases {
        assert_solves_exactly(&pool, f, x);
    }
}

// ---------------------------------------------------------------------------
// 6. Unit tests for the lattice machinery
// ---------------------------------------------------------------------------

#[test]
fn lattice_collapses_proportional_exponents() {
    use rug::Integer;
    // Rows [2], [3] generate the lattice ℤ (gcd 1).
    let rows = vec![vec![Integer::from(2)], vec![Integer::from(3)]];
    let (basis, pivots) = ring::lattice_basis(rows, 1);
    assert_eq!(basis.len(), 1);
    assert_eq!(basis[0][0], 1);
    assert_eq!(pivots, vec![0]);
    let combo = ring::int_combo(&basis, &pivots, &[Integer::from(-4)]).expect("in the lattice");
    assert_eq!(combo[0], -4);
}

#[test]
fn lattice_rejects_vectors_outside_it() {
    use rug::Integer;
    let rows = vec![vec![Integer::from(2), Integer::from(0)]];
    let (basis, pivots) = ring::lattice_basis(rows, 2);
    assert!(ring::int_combo(&basis, &pivots, &[Integer::from(1), Integer::from(0)]).is_none());
    assert!(ring::int_combo(&basis, &pivots, &[Integer::from(0), Integer::from(1)]).is_none());
}

#[test]
fn atom_decomposition_splits_rational_coefficients() {
    let (pool, x) = setup();
    // −3x/2 + 1
    let t = pool.mul(vec![pool.rational(-3_i32, 2_i32), x]);
    let e = pool.add(vec![t, pool.integer(1_i32)]);
    let d = ring::atom_decompose(e, &pool).expect("decomposable");
    assert_eq!(d.len(), 2);
    let one = pool.integer(1_i32);
    let cx = d.iter().find(|(a, _)| *a == x).expect("x atom").1.clone();
    let c1 = d
        .iter()
        .find(|(a, _)| *a == one)
        .expect("constant atom")
        .1
        .clone();
    assert_eq!(cx, rug::Rational::from((-3, 2)));
    assert_eq!(c1, rug::Rational::from(1));
}

// ---------------------------------------------------------------------------
// 7. Edge cases and regressions
// ---------------------------------------------------------------------------

#[test]
fn the_zero_integrand_integrates_to_zero() {
    let (pool, x) = setup();
    let got = assert_solves(&pool, pool.integer(0_i32), x);
    assert_eq!(got, pool.integer(0_i32));
}

#[test]
fn a_constant_integrand_integrates_to_a_multiple_of_x() {
    let (pool, x) = setup();
    // ∫5 dx = 5x.  Trivial, but it exercises the degenerate `Q = 1` ansatz,
    // where the monomial box is `{1, x}` and the log candidate set is `{x}`.
    let five = pool.integer(5_i32);
    let got = assert_solves_exactly(&pool, five, x);
    let d = crate::diff::diff(got, x, &pool)
        .expect("differentiable")
        .value;
    assert_eq!(crate::simplify::engine::simplify(d, &pool).value, five);
}

#[test]
fn a_rational_constant_integrand_stays_exact() {
    let (pool, x) = setup();
    let c = pool.rational(-3_i32, 7_i32);
    let got = assert_solves_exactly(&pool, c, x);
    let d = crate::diff::diff(got, x, &pool)
        .expect("differentiable")
        .value;
    assert_eq!(crate::simplify::engine::simplify(d, &pool).value, c);
}

#[test]
fn an_integrand_that_is_already_an_ansatz_atom_solves() {
    let (pool, x) = setup();
    // `∫exp(x) dx` — the integrand *is* the answer, so the rational part of
    // the ansatz has to carry it with a unit coefficient.
    let f = exp_of(&pool, x);
    let got = assert_solves_exactly(&pool, f, x);
    assert!(verify_antiderivative_exact(got, f, x, &pool));
}

#[test]
fn a_free_symbol_is_out_of_ring_rather_than_a_verdict() {
    let (pool, x) = setup();
    let y = pool.symbol("y", Domain::Real);
    let f = pool.mul(vec![y, x]);
    assert!(matches!(
        assert_declines(&pool, f, x),
        DeclineReason::UnsupportedIntegrand(_)
    ));
}

/// `exp(x)` and `exp(2x)` are the *same* generator, not two.
///
/// This is the exponential analogue of the `√x` / `√(4x)` defect found
/// elsewhere in this crate, where two spellings of one object were treated as
/// independent.  Here the lattice reduction collapses them, so
/// `∫exp(2x)/(exp(x)+1)` — which mentions both — is a one-generator problem.
#[test]
fn proportional_exponentials_are_one_generator_not_two() {
    let (pool, x) = setup();
    let e2 = exp_of(&pool, pool.mul(vec![pool.integer(2_i32), x]));
    let e1 = exp_of(&pool, x);
    let den = pool.add(vec![e1, pool.integer(1_i32)]);
    let f = pool.mul(vec![e2, inv(&pool, den)]);
    assert_solves(&pool, f, x);
}

/// `log(x)` and `log(x²)` are multiplicatively dependent modulo constants, so
/// they are not two independent coordinates.
///
/// The module may collapse them or decline; what it must never do is treat
/// them as independent and hand back an answer that has not been gated.
#[test]
fn a_logarithm_and_its_power_never_escape_the_gate() {
    let (pool, x) = setup();
    let l1 = log_of(&pool, x);
    let l2 = log_of(&pool, pool.pow(x, pool.integer(2_i32)));
    let f = pool.mul(vec![l1, l2]);
    match integrate_parallel_risch(f, x, &pool) {
        ParallelRischOutcome::Solved { antiderivative, .. } => assert!(
            verify_antiderivative_status(antiderivative, f, x, &pool).is_some(),
            "a dependent-generator answer escaped the gate"
        ),
        ParallelRischOutcome::Declined(_) => {}
    }
}

/// `x^(i64::MIN)` must decline, not spin.
///
/// `rf_pow` guarded its exponent with `k.abs() > MAX_POW`.  `i64::MIN.abs()`
/// overflows: in a release build it wraps back to `i64::MIN`, which compares
/// *below* the cap, so the guard passed — and the multiply loop underneath then
/// ran `k.unsigned_abs() = 2⁶³` times.  Reachable from user input.
#[test]
fn an_extreme_negative_exponent_declines_instead_of_hanging() {
    let (pool, x) = setup();
    let f = pool.pow(x, pool.integer(i64::MIN));
    assert!(matches!(
        assert_declines(&pool, f, x),
        DeclineReason::TooLarge(_) | DeclineReason::UnsupportedIntegrand(_)
    ));
}

/// A deeply nested integrand must decline, not exhaust the stack.
///
/// `collect`, `to_rf` and — before either of them — `simplify` all recurse over
/// the expression tree.  A stack overflow aborts the process: not catchable, and
/// strictly worse than a panic.  Before the guard, this input killed the test
/// binary with `SIGABRT`.
#[test]
fn deep_nesting_declines_instead_of_overflowing_the_stack() {
    let (pool, x) = setup();
    let mut e = x;
    for _ in 0..5000 {
        e = pool.add(vec![e, pool.integer(1_i32)]);
    }
    assert!(matches!(
        assert_declines(&pool, e, x),
        DeclineReason::TooLarge(_)
    ));
}

/// The depth guard must not mistake a wide, shallow, heavily shared expression
/// for a deep one — the pool is a DAG, and counting paths instead of depth
/// would reject `((x+1)·(x+1))·((x+1)·(x+1))·…` out of hand.
#[test]
fn the_depth_guard_measures_depth_not_paths() {
    let (pool, x) = setup();
    let mut e = pool.add(vec![x, pool.integer(1_i32)]);
    for _ in 0..40 {
        e = pool.mul(vec![e, e]);
    }
    // 2^40 paths, 41 levels deep.  The guard must terminate and must not fire.
    assert!(!ring::depth_exceeds(e, &pool, ring::MAX_DEPTH));
}

/// "The solver gave up" has to stay distinct from "the system has no
/// solution", and neither may become a verdict about the integrand.
#[test]
fn the_linear_solver_decline_is_distinct_and_still_not_a_verdict() {
    assert_ne!(DeclineReason::LinearSolver, DeclineReason::NoSolution);
    assert!(matches!(
        DeclineReason::LinearSolver.into_integration_error(),
        IntegrationError::NotImplemented(_)
    ));
}
