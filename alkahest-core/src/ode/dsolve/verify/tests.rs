//! Tests for the verification gate itself.
//!
//! These call [`residual_is_zero`] directly with hand-built candidates rather
//! than going through `dsolve`, because the point is to pin what the *gate*
//! does with a candidate — including candidates no implemented solving class
//! would ever produce.  Both directions are pinned: a correct solution of a
//! singular ODE must still verify, and a wrong candidate whose blow-up used to
//! be skipped must now be caught.

use super::*;
use crate::kernel::Domain;

/// `y`, `y'`, … named as `OdeInput` names them, plus an integration constant.
struct Fixture {
    pool: ExprPool,
    x: ExprId,
    y: ExprId,
    c1: ExprId,
}

impl Fixture {
    fn new() -> Self {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let c1 = pool.symbol("C1", Domain::Real);
        Fixture { pool, x, y, c1 }
    }

    fn neg(&self, e: ExprId) -> ExprId {
        self.pool.mul(vec![self.pool.integer(-1_i32), e])
    }

    /// `x − n/d`.
    fn x_minus(&self, n: i32, d: i32) -> ExprId {
        let r = self.pool.rational(n, d);
        self.pool.add(vec![self.x, self.neg(r)])
    }

    /// `√(1/2 − x)` — real for `x < 1/2`, NaN for the `0.61` and `0.79`
    /// samples, which is what makes an expression containing it "singular
    /// there" for the sampler.
    fn sqrt_half_minus_x(&self) -> ExprId {
        let half = self.pool.rational(1_i32, 2_i32);
        let arg = self.pool.add(vec![half, self.neg(self.x)]);
        self.pool.func("sqrt", vec![arg])
    }

    /// `eˣ·e⁻ˣ`, which is `1` but which `simplify` does not collapse.
    ///
    /// Multiplying a term by this is how these tests force the gate past its
    /// symbolic-zero branch and onto the numeric sampler — the same artefact
    /// the real corpus residuals carry (`x⁻¹·eˣ·e⁻ˣ − x⁻¹` for `y'' − y = 1/x`).
    fn one_that_does_not_simplify(&self) -> ExprId {
        let ex = self.pool.func("exp", vec![self.x]);
        let emx = self.pool.func("exp", vec![self.neg(self.x)]);
        self.pool.mul(vec![ex, emx])
    }
}

// ---------------------------------------------------------------------------
// The soundness direction: a wrong candidate must not be certified
// ---------------------------------------------------------------------------

/// The reproducer for the bug this module was restructured to fix.
///
/// The ODE is `y' = 0`, which is regular at every `x`; its general solution is
/// `y = C1`.  The candidate adds `√(x − ½)·(x − 0.61)²·(x − 0.79)²`, which is
///
/// * NaN at the samples `0.11, 0.27, 0.43` (negative radicand), and
/// * a double root at `0.61` and `0.79`, so the residual `y'` is exactly zero
///   at the two samples that *do* evaluate.
///
/// The old gate skipped the three NaN samples, saw `2 × 3 = 6` finite samples
/// all `≈ 0`, cleared its `≥ 6` bar and returned `Ok(())` — a wrong `y(x)`
/// certified.  The candidate blows up where the ODE is perfectly regular, so
/// the classifier now books those samples as blow-ups and refuses.
#[test]
fn wrong_candidate_blowing_up_at_a_regular_point_is_rejected() {
    let f = Fixture::new();
    let (input, yp) = OdeInput::first_order(f.x, f.y, &f.pool);
    let input = input.with_equation(yp);

    let root = f.pool.func("sqrt", vec![f.x_minus(-1, -2)]);
    let sq = |n: i32, d: i32| f.pool.pow(f.x_minus(n, d), f.pool.integer(2_i32));
    let wrong = simp(
        f.pool
            .add(vec![f.c1, f.pool.mul(vec![root, sq(61, 100), sq(79, 100)])]),
        &f.pool,
    );

    let (residual, derivs) = build_residual(&input, wrong, &f.pool).expect("residual builds");
    let report = numeric_report(&input, residual, &derivs, &[f.c1], &f.pool);

    // The shape of the trap: exactly six agreeing samples, no disagreement, and
    // nine samples that the old code discarded.
    assert_eq!(report.agree, 6, "report: {report}");
    assert_eq!(report.disagree, 0, "report: {report}");
    assert_eq!(report.blowup_at_regular_point, 9, "report: {report}");
    assert_eq!(report.skipped_singular_ode, 0, "report: {report}");
    assert!(!report.certifies(), "report: {report}");

    residual_is_zero(&input, wrong, &[f.c1], &f.pool)
        .expect_err("a candidate that blows up where the ODE is regular must not certify");
}

/// A candidate that simply disagrees at finite samples is still rejected — the
/// restructure must not have loosened the ordinary path.  `y = C1·eˣ` is not a
/// solution of `y' = 2y`.
#[test]
fn plain_finite_disagreement_is_still_rejected() {
    let f = Fixture::new();
    let (input, yp) = OdeInput::first_order(f.x, f.y, &f.pool);
    // y' − 2·y·(eˣ·e⁻ˣ) = 0, i.e. y' = 2y, spelled so the symbolic branch misses.
    let two_y = f.pool.mul(vec![
        f.pool.integer(-2_i32),
        f.y,
        f.one_that_does_not_simplify(),
    ]);
    let input = input.with_equation(f.pool.add(vec![yp, two_y]));

    let cand = simp(
        f.pool.mul(vec![f.c1, f.pool.func("exp", vec![f.x])]),
        &f.pool,
    );
    let (residual, derivs) = build_residual(&input, cand, &f.pool).expect("residual builds");
    let report = numeric_report(&input, residual, &derivs, &[f.c1], &f.pool);
    assert_eq!(report.agree, 0, "report: {report}");
    assert_eq!(report.disagree, 15, "report: {report}");
    assert!(!report.certifies(), "report: {report}");
}

// ---------------------------------------------------------------------------
// The capability direction: a correct solution must still be certified
// ---------------------------------------------------------------------------

/// A correct solution of an ODE that is *singular over part of the sample
/// range* must still verify.
///
/// The equation is `y' − y − √(½ − x)·eˣ·e⁻ˣ + √(½ − x) = 0`.  The last two
/// terms cancel as functions (`eˣ·e⁻ˣ = 1`), so the equation is `y' = y` and
/// `y = C1·eˣ` is exactly right — but as *written* the equation does not
/// evaluate at all for `x > ½`, so the samples `0.61` and `0.79` carry no
/// information about the candidate and must be skipped, not counted against it.
#[test]
fn correct_solution_of_a_singular_ode_still_verifies() {
    let f = Fixture::new();
    let (input, yp) = OdeInput::first_order(f.x, f.y, &f.pool);
    let root = f.sqrt_half_minus_x();
    let eq = f.pool.add(vec![
        yp,
        f.neg(f.y),
        f.neg(f.pool.mul(vec![root, f.one_that_does_not_simplify()])),
        root,
    ]);
    let input = input.with_equation(eq);

    let cand = simp(
        f.pool.mul(vec![f.c1, f.pool.func("exp", vec![f.x])]),
        &f.pool,
    );

    // The premise: the ODE really is unevaluable past the branch point, and
    // fine before it.
    assert!(ode_is_regular_at(&input, 0.43, &f.pool));
    assert!(!ode_is_regular_at(&input, 0.61, &f.pool));

    let (residual, derivs) = build_residual(&input, cand, &f.pool).expect("residual builds");
    let report = numeric_report(&input, residual, &derivs, &[f.c1], &f.pool);
    assert_eq!(report.agree, 9, "report: {report}");
    assert_eq!(report.disagree, 0, "report: {report}");
    assert_eq!(report.blowup_at_regular_point, 0, "report: {report}");
    assert_eq!(report.skipped_singular_ode, 6, "report: {report}");
    assert!(report.certifies(), "report: {report}");

    residual_is_zero(&input, cand, &[f.c1], &f.pool)
        .expect("a correct solution of a singular ODE must still verify");
}

/// When the candidate blows up *and* the ODE is singular at the same sample,
/// the sample is still information-free and must stay a skip.
///
/// `y' = y·eˣ·e⁻ˣ/√(½ − x)` has the correct solution `y = C1·e^{−2√(½ − x)}`.
/// Both the equation and the solution are NaN for `x > ½`, so the ordering in
/// [`classify_nonfinite`] — singular ODE first — is what keeps this verifying.
#[test]
fn candidate_blowup_where_the_ode_is_also_singular_stays_a_skip() {
    let f = Fixture::new();
    let (input, yp) = OdeInput::first_order(f.x, f.y, &f.pool);
    let root = f.sqrt_half_minus_x();
    let inv_root = f.pool.pow(root, f.pool.integer(-1_i32));
    let rhs = f
        .pool
        .mul(vec![f.y, f.one_that_does_not_simplify(), inv_root]);
    let input = input.with_equation(f.pool.add(vec![yp, f.neg(rhs)]));

    // y = C1·exp(−2·√(½ − x))
    let expo = f.pool.mul(vec![f.pool.integer(-2_i32), root]);
    let cand = simp(
        f.pool.mul(vec![f.c1, f.pool.func("exp", vec![expo])]),
        &f.pool,
    );

    assert!(!ode_is_regular_at(&input, 0.61, &f.pool));

    let (residual, derivs) = build_residual(&input, cand, &f.pool).expect("residual builds");
    let report = numeric_report(&input, residual, &derivs, &[f.c1], &f.pool);
    assert_eq!(report.disagree, 0, "report: {report}");
    assert_eq!(report.blowup_at_regular_point, 0, "report: {report}");
    assert_eq!(report.skipped_singular_ode, 6, "report: {report}");
    assert_eq!(report.agree, 9, "report: {report}");

    residual_is_zero(&input, cand, &[f.c1], &f.pool)
        .expect("correct solution, ODE singular past the branch point");
}

// ---------------------------------------------------------------------------
// The regularity probe
// ---------------------------------------------------------------------------

/// `ode_is_regular_at` must answer about the *equation*, not about one state:
/// a single unlucky probe value may make a perfectly regular equation
/// unevaluable, so one finite probe is enough to call it regular.
#[test]
fn regularity_probe_ignores_an_unlucky_state() {
    let f = Fixture::new();
    let (input, yp) = OdeInput::first_order(f.x, f.y, &f.pool);
    // y' − log(y) = 0: NaN for the negative probe, finite for the positive ones.
    let eq = f.pool.add(vec![yp, f.neg(f.pool.func("log", vec![f.y]))]);
    let input = input.with_equation(eq);
    assert!(ode_is_regular_at(&input, 0.43, &f.pool));

    // y' − log(½ − x) = 0 is genuinely unevaluable past the branch point, for
    // every state.
    let (input2, yp2) = OdeInput::first_order(f.x, f.y, &f.pool);
    let half = f.pool.rational(1_i32, 2_i32);
    let arg = f.pool.add(vec![half, f.neg(f.x)]);
    let eq2 = f.pool.add(vec![yp2, f.neg(f.pool.func("log", vec![arg]))]);
    let input2 = input2.with_equation(eq2);
    assert!(ode_is_regular_at(&input2, 0.43, &f.pool));
    assert!(!ode_is_regular_at(&input2, 0.61, &f.pool));
}

/// An equation the sampler cannot evaluate at all (an unknown special function)
/// is never called regular, so a candidate is never condemned on the strength
/// of a probe that did not run.
#[test]
fn regularity_probe_declines_on_an_unknown_construct() {
    let f = Fixture::new();
    let (input, yp) = OdeInput::first_order(f.x, f.y, &f.pool);
    let ei = f.pool.func("Ei", vec![f.x]);
    let input = input.with_equation(f.pool.add(vec![yp, f.neg(ei)]));
    assert!(!ode_is_regular_at(&input, 0.43, &f.pool));
}
