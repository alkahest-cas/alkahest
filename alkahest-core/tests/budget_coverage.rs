//! G8 — the active `Budget` covers the expensive algebraic paths.
//!
//! Each engine below used to run to completion whatever budget was active: a
//! Gröbner basis, polynomial `solve`, primary decomposition, triangularization,
//! Rosenfeld–Gröbner, homotopy continuation, the parametric Gröbner engine and
//! two-variable CAD. Each now polls the budget in its unbounded loops and
//! reports a trip as an error — with the trip itself left in
//! `budget::take_trip` where the error enum is exhaustive — and never returns a
//! partial result as a success.
//!
//! A step budget of zero is the deterministic way to trip at the first
//! interrupt point; the wall-clock tests use the report's hard inputs with a
//! generous tolerance for loaded CI machines.
#![cfg(feature = "groebner")]

use alkahest_cas::budget::{self, Budget, BudgetError, BudgetTrip};
use alkahest_cas::kernel::{Domain, ExprId, ExprPool};
use alkahest_cas::poly::groebner::{GbPoly, GroebnerBasis, MonomialOrder, ParamGroebnerBasis};
use alkahest_cas::{
    decide_expr, expr_to_gbpoly, expr_to_param_gbpoly, primary_decomposition,
    rosenfeld_groebner_algebraic, solve_numerical, solve_polynomial_system, triangularize,
    CadError, HomotopyError, HomotopyOpts, ParamGroebnerError, PrimaryDecompositionError,
    SolverError,
};
use std::time::{Duration, Instant};

fn steps_trip() -> bool {
    matches!(
        budget::take_trip(),
        Some(BudgetTrip::Budget(BudgetError::Steps { .. }))
    )
}

fn int(pool: &ExprPool, n: i64) -> ExprId {
    pool.integer(n)
}

fn pow(pool: &ExprPool, b: ExprId, e: i64) -> ExprId {
    pool.pow(b, pool.integer(e))
}

/// `x⁷ + y⁵ − 3xy − 1, y⁷ − x³y + 2` — the report's `solve` repro (9.1 s
/// against a 300 ms budget on 3.12.0).
fn hard_system(pool: &ExprPool) -> (Vec<ExprId>, Vec<ExprId>) {
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let e1 = pool.add(vec![
        pow(pool, x, 7),
        pow(pool, y, 5),
        pool.mul(vec![int(pool, -3), x, y]),
        int(pool, -1),
    ]);
    let e2 = pool.add(vec![
        pow(pool, y, 7),
        pool.mul(vec![int(pool, -1), pow(pool, x, 3), y]),
        int(pool, 2),
    ]);
    (vec![e1, e2], vec![x, y])
}

fn small_system(pool: &ExprPool) -> (Vec<ExprId>, Vec<ExprId>) {
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    // x² + y² − 1, x − y
    let e1 = pool.add(vec![pow(pool, x, 2), pow(pool, y, 2), int(pool, -1)]);
    let e2 = pool.add(vec![x, pool.mul(vec![int(pool, -1), y])]);
    (vec![e1, e2], vec![x, y])
}

#[test]
fn solve_reports_a_budget_trip_and_no_solutions() {
    let pool = ExprPool::new();
    let (eqs, vars) = small_system(&pool);
    let _g = budget::enter(Budget::new().with_max_steps(0));
    let r = solve_polynomial_system(eqs, vars, &pool);
    assert!(matches!(r, Err(SolverError::NotPolynomial(ref m)) if m.contains("budget")));
    assert!(steps_trip());
}

#[test]
fn solve_still_solves_under_a_generous_budget() {
    let pool = ExprPool::new();
    let (eqs, vars) = small_system(&pool);
    let _g = budget::enter(Budget::new().with_wall(Duration::from_secs(600)));
    let r = solve_polynomial_system(eqs, vars, &pool).expect("solves");
    assert!(matches!(r, alkahest_cas::SolutionSet::Finite(ref s) if s.len() == 2));
    assert!(budget::take_trip().is_none());
}

#[test]
fn solve_hard_system_stops_near_the_wall_budget() {
    let pool = ExprPool::new();
    let (eqs, vars) = hard_system(&pool);
    let _g = budget::enter(Budget::new().with_wall(Duration::from_millis(300)));
    let t = Instant::now();
    let r = solve_polynomial_system(eqs, vars, &pool);
    let elapsed = t.elapsed();
    assert!(
        r.is_err(),
        "the degree-49 system cannot be solved in 300 ms"
    );
    assert!(matches!(
        budget::take_trip(),
        Some(BudgetTrip::Budget(BudgetError::WallClock { .. }))
    ));
    // 300 ms budget; generous (100x) so a loaded or sanitizer run cannot trip it.
    assert!(elapsed < Duration::from_secs(30), "took {elapsed:?}");
}

#[test]
fn homotopy_reports_a_budget_trip() {
    let pool = ExprPool::new();
    let (eqs, vars) = hard_system(&pool);
    let _g = budget::enter(Budget::new().with_max_steps(0));
    let r = solve_numerical(&eqs, &vars, &pool, &HomotopyOpts::default());
    assert!(matches!(r, Err(HomotopyError::TrackerFailed(_))), "{r:?}");
    assert!(steps_trip());
}

#[test]
fn triangularize_reports_a_budget_trip() {
    let pool = ExprPool::new();
    let (eqs, vars) = small_system(&pool);
    let _g = budget::enter(Budget::new().with_max_steps(0));
    let r = triangularize(eqs, vars, &pool);
    assert!(matches!(r, Err(SolverError::NotPolynomial(_))));
    assert!(steps_trip());
}

fn gb_system(pool: &ExprPool) -> Vec<GbPoly> {
    let (eqs, vars) = small_system(pool);
    eqs.iter()
        .map(|&e| expr_to_gbpoly(e, &vars, pool).unwrap())
        .collect()
}

#[test]
fn primary_decomposition_reports_a_budget_trip() {
    let pool = ExprPool::new();
    let gens = gb_system(&pool);
    {
        let _g = budget::enter(Budget::new().with_max_steps(0));
        let r = primary_decomposition(gens.clone(), MonomialOrder::Lex);
        assert!(
            matches!(r, Err(PrimaryDecompositionError::Factorization(_))),
            "{r:?}"
        );
        assert!(steps_trip());
    }
    // And still decomposes without one: x = y, 2y² = 1 is one prime over ℚ.
    let comps = primary_decomposition(gens, MonomialOrder::Lex).expect("decomposes");
    assert_eq!(comps.len(), 1);
}

#[test]
fn rosenfeld_groebner_reports_a_budget_trip() {
    let pool = ExprPool::new();
    let gens = gb_system(&pool);
    let _g = budget::enter(Budget::new().with_max_steps(0));
    let r = rosenfeld_groebner_algebraic(gens, MonomialOrder::Lex);
    assert!(r.is_err());
    assert!(steps_trip());
}

#[test]
fn parametric_groebner_reports_a_budget_trip() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let a = pool.symbol("a", Domain::Real);
    // a·x − y, x + y − 1  over ℚ(a)
    let e1 = pool.add(vec![
        pool.mul(vec![a, x]),
        pool.mul(vec![int(&pool, -1), y]),
    ]);
    let e2 = pool.add(vec![x, y, int(&pool, -1)]);
    let gens: Vec<_> = [e1, e2]
        .iter()
        .map(|&e| expr_to_param_gbpoly(e, &[x, y], &[a], &pool).unwrap())
        .collect();
    {
        let _g = budget::enter(Budget::new().with_max_steps(0));
        let r = ParamGroebnerBasis::compute(gens.clone(), MonomialOrder::Lex);
        assert!(
            matches!(
                r,
                Err(ParamGroebnerError::Budget(BudgetTrip::Budget(
                    BudgetError::Steps { .. }
                )))
            ),
            "{r:?}"
        );
    }
    assert!(ParamGroebnerBasis::compute(gens, MonomialOrder::Lex).is_ok());
}

#[test]
fn two_variable_decide_reports_a_budget_trip() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    // ∃x ∃y. x² + y² < 1  (true)
    let body = pool.pred_lt(
        pool.add(vec![pow(&pool, x, 2), pow(&pool, y, 2)]),
        int(&pool, 1),
    );
    let phi = pool.exists(x, pool.exists(y, body));
    {
        let _g = budget::enter(Budget::new().with_max_steps(0));
        let r = decide_expr(phi, &pool);
        assert!(matches!(r, Err(CadError::Unsupported(_))), "{r:?}");
        assert!(steps_trip());
    }
    assert!(decide_expr(phi, &pool).expect("decides").truth);
}

/// The infallible Gröbner constructor is unchanged: it neither consults the
/// budget nor leaves a trip behind, because it has no way to report one.
#[test]
fn infallible_groebner_ignores_the_budget() {
    let pool = ExprPool::new();
    let gens = gb_system(&pool);
    let _g = budget::enter(Budget::new().with_max_steps(0));
    let gb = GroebnerBasis::compute(gens.clone(), MonomialOrder::Lex);
    assert_eq!(gb.len(), 2);
    assert!(budget::take_trip().is_none());
    assert!(GroebnerBasis::try_compute(gens, MonomialOrder::Lex).is_err());
}
