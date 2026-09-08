//! Tests for the linear-system solver.
//!
//! Every case asserts the *whole* system verifies, not just that a shape came
//! back: `dsolve_system` already refuses an unverified candidate, so the
//! assertions here are about which systems reach an answer at all and about the
//! branch bookkeeping that answer carries.

use super::*;
use crate::deriv::SideCondition;
use crate::kernel::{Domain, ExprPool};
use crate::parse::parse;
use std::collections::HashMap;

/// Build `d(state_i)/dt = rhs_i` from parsed sources over a shared symbol table.
fn build(states: &[&str], rhs: &[&str], params: &[&str]) -> (ExprPool, ODE, Vec<ExprId>) {
    let pool = ExprPool::new();
    let t = pool.symbol("t", Domain::Real);
    let mut syms: HashMap<String, ExprId> = HashMap::new();
    syms.insert("t".to_owned(), t);
    let state_ids: Vec<ExprId> = states
        .iter()
        .map(|s| {
            let id = pool.symbol(*s, Domain::Real);
            syms.insert((*s).to_owned(), id);
            id
        })
        .collect();
    let param_ids: Vec<ExprId> = params
        .iter()
        .map(|s| {
            let id = pool.symbol(*s, Domain::Real);
            syms.insert((*s).to_owned(), id);
            id
        })
        .collect();
    let rhs_ids: Vec<ExprId> = rhs
        .iter()
        .map(|src| parse(src, &pool, &mut syms).expect("rhs parses"))
        .collect();
    let ode = ODE::new(state_ids, rhs_ids, t, &pool).expect("well-formed system");
    (pool, ode, param_ids)
}

fn nonzero_conditions(sol: &SystemSolution, pool: &ExprPool) -> Vec<String> {
    sol.side_conditions
        .iter()
        .map(|c| match c {
            SideCondition::NonZero(id) => pool.display(*id).to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Numeric coefficients
// ---------------------------------------------------------------------------

#[test]
fn scalar_decay_is_an_exponential() {
    let (pool, ode, _) = build(&["x"], &["-3*x"], &[]);
    let sol = dsolve_system(&ode, &pool).expect("x' = -3x solves");
    assert_eq!(sol.y_of_t.len(), 1);
    assert_eq!(sol.constants.len(), 1);
    assert!(sol.side_conditions.is_empty());
}

#[test]
fn two_by_two_with_distinct_integer_eigenvalues() {
    // x' = x + 2y, y' = 3x + 2y — eigenvalues 4 and -1.
    let (pool, ode, _) = build(&["x", "y"], &["x + 2*y", "3*x + 2*y"], &[]);
    let sol = dsolve_system(&ode, &pool).expect("solves");
    assert_eq!(sol.constants.len(), 2);
    assert!(
        sol.side_conditions.is_empty(),
        "integer eigenvalues leave nothing undecided, got {:?}",
        nonzero_conditions(&sol, &pool)
    );
}

#[test]
fn rotation_has_a_complex_spectrum_and_still_verifies() {
    // x' = -y, y' = x — eigenvalues ±i.  The answer is written with complex
    // exponentials; the gate evaluates it over ℂ, which is the point.
    let (pool, ode, _) = build(&["x", "y"], &["-1*y", "x"], &[]);
    let sol = dsolve_system(&ode, &pool).expect("solves");
    assert_eq!(sol.constants.len(), 2);
}

/// The case the module docs promise Putzer handles without Jordan machinery.
#[test]
fn defective_matrix_needs_no_jordan_form() {
    // x' = 2x + y, y' = 2y — a single Jordan block for λ = 2.
    let (pool, ode, _) = build(&["x", "y"], &["2*x + y", "2*y"], &[]);
    let sol = dsolve_system(&ode, &pool).expect("defective system solves");
    assert_eq!(sol.constants.len(), 2);
    assert!(sol.side_conditions.is_empty());
    // e^{At} = [[e^{2t}, t·e^{2t}], [0, e^{2t}]]: the off-diagonal entry must
    // carry the secular `t` factor, which is exactly what a distinct-root
    // formula would not produce.
    let off = pool.display(sol.fundamental_matrix[0][1]).to_string();
    assert!(off.contains('t'), "expected a t·e^(2t) entry, got {off}");
    assert!(
        matches!(pool.get(sol.fundamental_matrix[1][0]), ExprData::Integer(n) if n.0 == 0),
        "lower-left entry of a triangular exponential must be 0"
    );
}

#[test]
fn triple_jordan_block_solves() {
    // A single 3×3 Jordan block at λ = 1: x' = x + y, y' = y + z, z' = z.
    let (pool, ode, _) = build(&["x", "y", "z"], &["x + y", "y + z", "z"], &[]);
    let sol = dsolve_system(&ode, &pool).expect("triple block solves");
    assert_eq!(sol.constants.len(), 3);
    let corner = pool.display(sol.fundamental_matrix[0][2]).to_string();
    assert!(
        corner.contains("t^2") || corner.contains("t ^ 2") || corner.contains("(t * t)"),
        "the corner of a 3×3 Jordan block carries t²/2, got {corner}"
    );
}

/// `pi` is an ordinary symbol in this crate, so a gate that collects free
/// symbols and binds each to a sample value binds **π to 1.7** — and every
/// expression whose correctness depends on π being π then disagrees at every
/// sample.  The casus irreducibilis is where that bites: the three real
/// eigenvalues of a `λ³ − 3λ + 1` companion matrix are
/// `2√(−p/3)·cos((acos c + 2πk)/3)`, correct only at the real π, and the system
/// built on them used to be refused with `VerificationFailed`.
#[test]
fn casus_irreducibilis_eigenvalues_are_not_refused_over_a_sampled_pi() {
    // Companion matrix of λ³ − 3λ + 1 (Δ < 0: three distinct real roots).
    // x' = -z, y' = x + 3z, z' = y.
    let (pool, ode, _) = build(&["x", "y", "z"], &["-1*z", "x + 3*z", "y"], &[]);
    let sol = dsolve_system(&ode, &pool).expect("the trigonometric eigenvalues verify");
    assert_eq!(sol.constants.len(), 3);
    let shown = pool.display(sol.y_of_t[0]).to_string();
    assert!(
        shown.contains("pi") || shown.contains('π') || shown.contains("acos"),
        "expected the trigonometric root form, got {shown}"
    );
}

#[test]
fn forced_system_closes_by_variation_of_parameters() {
    // x' = -x + 1, y' = x - 2y.
    let (pool, ode, _) = build(&["x", "y"], &["-1*x + 1", "x - 2*y"], &[]);
    let sol = dsolve_system(&ode, &pool).expect("forced system solves");
    assert_eq!(sol.method, "linear_system_putzer_variation_of_parameters");
    assert_eq!(sol.constants.len(), 2);
}

// ---------------------------------------------------------------------------
// Symbolic coefficients
// ---------------------------------------------------------------------------

/// The headline case: the two-compartment pharmacokinetic model.
#[test]
fn two_compartment_pk_model_with_symbolic_rates() {
    let (pool, ode, _) = build(&["x", "y"], &["-1*ka*x", "ka*x - ke*y"], &["ka", "ke"]);
    let sol = dsolve_system(&ode, &pool).expect("PK model solves");
    assert_eq!(sol.constants.len(), 2);
    let conds = nonzero_conditions(&sol, &pool);
    assert_eq!(
        conds.len(),
        1,
        "exactly one confluence (ka = ke) is undecided, got {conds:?}"
    );
    let c = &conds[0];
    assert!(
        c.contains("ka") && c.contains("ke"),
        "the condition must name both rates, got {c}"
    );
    assert!(
        sol.notes.iter().any(|n| n.contains("confluent")),
        "the ka = ke case must be named in prose"
    );
    // The first compartment decays purely: x(t) = C1·e^{−ka·t}, with no `ke`.
    let x = pool.display(sol.y_of_t[0]).to_string();
    assert!(
        !x.contains("ke"),
        "the donor compartment cannot depend on the eliminating rate, got {x}"
    );
}

/// Asserting the confluence away removes the side condition.
#[test]
fn asserted_distinct_rates_carry_no_side_condition() {
    let (pool, ode, params) = build(&["x", "y"], &["-1*ka*x", "ka*x - ke*y"], &["ka", "ke"]);
    let (ka, ke) = (params[0], params[1]);
    // ka − ke ≠ 0.  The eigenvalue difference the solver forms is ±(ka − ke),
    // and a fact pinning it up to a constant multiple settles either spelling.
    let diff = pool.add(vec![ka, pool.mul(vec![pool.integer(-1_i32), ke])]);
    let zero = pool.integer(0_i32);
    let mut assumptions = AssumptionContext::new();
    assumptions
        .refine(pool.pred_ne(diff, zero), &pool)
        .expect("ka ≠ ke is satisfiable");
    let sol = dsolve_system_with(&ode, &assumptions, &pool).expect("solves");
    assert!(
        sol.side_conditions.is_empty(),
        "the caller settled the only branch, got {:?}",
        nonzero_conditions(&sol, &pool)
    );
}

#[test]
fn symbolic_two_by_two_general_matrix() {
    let (pool, ode, _) = build(
        &["x", "y"],
        &["a*x + b*y", "c*x + d*y"],
        &["a", "b", "c", "d"],
    );
    let sol = dsolve_system(&ode, &pool).expect("general symbolic 2×2 solves");
    assert_eq!(sol.constants.len(), 2);
    assert!(
        !sol.side_conditions.is_empty(),
        "the discriminant of a fully symbolic 2×2 is undecided and must be reported"
    );
}

#[test]
fn symbolic_three_compartment_chain() {
    // A triangular 3-compartment chain: the spectrum is the diagonal, so no
    // Cardano radicals appear.
    let (pool, ode, _) = build(
        &["x", "y", "z"],
        &["-1*k1*x", "k1*x - k2*y", "k2*y - k3*z"],
        &["k1", "k2", "k3"],
    );
    let sol = dsolve_system(&ode, &pool).expect("3-compartment chain solves");
    assert_eq!(sol.constants.len(), 3);
    assert!(
        !sol.side_conditions.is_empty(),
        "the k_i = k_j confluences must be reported"
    );
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn a_nonlinear_system_is_refused_not_linearised() {
    let (pool, ode, _) = build(&["x", "y"], &["x*y", "y"], &[]);
    let err = dsolve_system(&ode, &pool).expect_err("x' = xy is not linear");
    assert!(matches!(err, DsolveSystemError::NotLinear(_)), "{err}");
}

#[test]
fn a_time_varying_coefficient_is_refused() {
    let (pool, ode, _) = build(&["x", "y"], &["t*x", "y"], &[]);
    let err = dsolve_system(&ode, &pool).expect_err("y' = A(t)y is out of scope");
    assert!(
        matches!(err, DsolveSystemError::NonConstantCoefficient(_)),
        "{err}"
    );
}

#[test]
fn every_error_carries_its_registered_code() {
    use crate::errors::AlkahestError;
    for (e, code) in [
        (DsolveSystemError::NotLinear(String::new()), "E-ODE-030"),
        (
            DsolveSystemError::NonConstantCoefficient(String::new()),
            "E-ODE-031",
        ),
        (
            DsolveSystemError::UnsupportedSpectrum(String::new()),
            "E-ODE-032",
        ),
        (DsolveSystemError::Unsupported(String::new()), "E-ODE-033"),
        (
            DsolveSystemError::VerificationFailed(String::new()),
            "E-ODE-034",
        ),
    ] {
        assert_eq!(e.code(), code);
        assert!(crate::errors::codes::REGISTRY
            .iter()
            .any(|s| s.code == code));
    }
}
