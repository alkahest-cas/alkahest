use rug::Rational;

use super::*;
use crate::budget::{self, Budget, BudgetError};
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprPool};

fn q(n: i64) -> Rational {
    Rational::from(n)
}

fn r(n: i64, d: i64) -> Rational {
    Rational::from((n, d))
}

fn qv(v: &[i64]) -> Vec<Rational> {
    v.iter().map(|&n| q(n)).collect()
}

fn optimal(lp: &LinearProgram) -> OptimalCertificate {
    match linprog(lp).unwrap() {
        LpOutcome::Optimal(c) => {
            c.verify(lp).unwrap();
            c
        }
        other => panic!("expected optimal, got {other:?}"),
    }
}

#[test]
fn textbook_maximisation_with_duals() {
    // max x + y  s.t. x + 2y ≤ 4, 3x + y ≤ 6, x, y ≥ 0  ⇒ (8/5, 6/5), 14/5.
    let mut lp = LinearProgram::new(qv(&[1, 1]));
    lp.sense = Sense::Maximize;
    lp.a_ub = vec![qv(&[1, 2]), qv(&[3, 1])];
    lp.b_ub = qv(&[4, 6]);
    let c = optimal(&lp);
    assert_eq!(c.x, vec![r(8, 5), r(6, 5)]);
    assert_eq!(c.objective, r(14, 5));
    // Max-sense duals are ≥ 0 on ≤ rows: y = (2/5, 1/5).
    assert_eq!(c.y_ub, vec![r(2, 5), r(1, 5)]);
}

#[test]
fn minimisation_matches_negated_maximisation() {
    let mut lp = LinearProgram::new(qv(&[-1, -1]));
    lp.a_ub = vec![qv(&[1, 2]), qv(&[3, 1])];
    lp.b_ub = qv(&[4, 6]);
    let c = optimal(&lp);
    assert_eq!(c.objective, r(-14, 5));
    assert_eq!(c.y_ub, vec![r(-2, 5), r(-1, 5)]);
}

#[test]
fn infeasible_has_farkas_vector() {
    // x + y ≤ 1, x + y ≥ 3 (as −x − y ≤ −3), x, y ≥ 0.
    let mut lp = LinearProgram::new(qv(&[1, 1]));
    lp.a_ub = vec![qv(&[1, 1]), qv(&[-1, -1])];
    lp.b_ub = qv(&[1, -3]);
    match linprog(&lp).unwrap() {
        LpOutcome::Infeasible(f) => {
            f.verify(&lp).unwrap();
            assert!(f.gap(&lp) < 0);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn crossed_bounds_are_infeasible() {
    let mut lp = LinearProgram::new(qv(&[1]));
    lp.bounds = vec![VarBound::new(Some(q(2)), Some(q(1)))];
    assert!(matches!(linprog(&lp).unwrap(), LpOutcome::Infeasible(_)));
}

#[test]
fn equality_infeasibility_with_free_variables() {
    let mut lp = LinearProgram::new(qv(&[0, 0]));
    lp.bounds = vec![VarBound::free(); 2];
    lp.a_eq = vec![qv(&[1, 1]), qv(&[2, 2])];
    lp.b_eq = qv(&[1, 3]);
    match linprog(&lp).unwrap() {
        LpOutcome::Infeasible(f) => f.verify(&lp).unwrap(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn unbounded_has_ray() {
    // min −x − y  s.t.  x − y ≤ 1, x, y ≥ 0.
    let mut lp = LinearProgram::new(qv(&[-1, -1]));
    lp.a_ub = vec![qv(&[1, -1])];
    lp.b_ub = qv(&[1]);
    match linprog(&lp).unwrap() {
        LpOutcome::Unbounded(u) => {
            u.verify(&lp).unwrap();
            assert!(dot(&lp.c, &u.direction) < 0);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn unconstrained_free_variable_is_unbounded() {
    let mut lp = LinearProgram::new(qv(&[1]));
    lp.bounds = vec![VarBound::free()];
    match linprog(&lp).unwrap() {
        LpOutcome::Unbounded(u) => assert_eq!(u.direction, vec![q(-1)]),
        other => panic!("{other:?}"),
    }
}

#[test]
fn beale_cycling_example_terminates() {
    // Beale (1955): cycles under Dantzig's rule; optimum −5/4 at (1, 0, 1, 0).
    let mut lp = LinearProgram::new(vec![r(-3, 4), q(20), r(-1, 2), q(6)]);
    lp.a_ub = vec![
        vec![r(1, 4), q(-8), q(-1), q(9)],
        vec![r(1, 2), q(-12), r(-1, 2), q(3)],
        vec![q(0), q(0), q(1), q(0)],
    ];
    lp.b_ub = qv(&[0, 0, 1]);
    let c = optimal(&lp);
    assert_eq!(c.objective, r(-5, 4));
    assert_eq!(c.x, vec![q(1), q(0), q(1), q(0)]);
}

#[test]
fn kuhn_cycling_example_terminates() {
    // Kuhn's example (Bazaraa et al.): degenerate at the origin.
    let mut lp = LinearProgram::new(qv(&[-2, -3, 1, 12]));
    lp.a_ub = vec![qv(&[-2, -9, 1, 9]), vec![r(1, 3), q(1), r(-1, 3), q(-2)]];
    lp.b_ub = qv(&[0, 0]);
    // Unbounded: (x1, x2, x3, x4) = t(…); either way the certificate checks.
    linprog(&lp).unwrap().verify(&lp).unwrap();
}

#[test]
fn degenerate_and_redundant_rows() {
    let mut lp = LinearProgram::new(qv(&[0, 0, -1]));
    lp.a_eq = vec![qv(&[1, 1, 1]), qv(&[2, 2, 2])];
    lp.b_eq = qv(&[1, 2]);
    lp.a_ub = vec![qv(&[1, 0, 0]), qv(&[1, 0, 0])];
    lp.b_ub = qv(&[1, 1]);
    let c = optimal(&lp);
    assert_eq!(c.objective, q(-1));
}

#[test]
fn general_bounds_and_free_variables() {
    // min x − 2y + z, −3 ≤ x ≤ 5, y ≤ 2 (no lower), z free, x + z ≥ 1, z ≤ 4
    // as rows; x − y = 0.
    let mut lp = LinearProgram::new(qv(&[1, -2, 1]));
    lp.bounds = vec![
        VarBound::new(Some(q(-3)), Some(q(5))),
        VarBound::new(None, Some(q(2))),
        VarBound::free(),
    ];
    lp.a_ub = vec![qv(&[-1, 0, -1]), qv(&[0, 0, 1])];
    lp.b_ub = qv(&[-1, 4]);
    lp.a_eq = vec![qv(&[1, -1, 0])];
    lp.b_eq = qv(&[0]);
    let c = optimal(&lp);
    // x = y ≤ 2; objective x − 2x + z = z − x with z ≥ 1 − x ⇒ 1 − 2x ⇒ x = 2.
    assert_eq!(c.objective, q(-3));
    assert_eq!(c.x[0], q(2));
}

#[test]
fn no_variables_or_rows() {
    let lp = LinearProgram::new(vec![]);
    assert_eq!(optimal(&lp).objective, q(0));
    let mut lp = LinearProgram::new(vec![]);
    lp.a_ub = vec![vec![]];
    lp.b_ub = qv(&[-1]);
    assert!(matches!(linprog(&lp).unwrap(), LpOutcome::Infeasible(_)));
}

#[test]
fn shape_errors_are_coded() {
    let mut lp = LinearProgram::new(qv(&[1, 1]));
    lp.a_ub = vec![qv(&[1])];
    lp.b_ub = qv(&[1]);
    assert_eq!(linprog(&lp).unwrap_err().code(), "E-LP-001");
    for code in [
        "E-LP-001", "E-LP-002", "E-LP-003", "E-LP-004", "E-LP-005", "E-LP-006",
    ] {
        assert!(
            crate::errors::codes::REGISTRY
                .iter()
                .any(|s| s.code == code),
            "{code} missing from REGISTRY"
        );
    }
}

#[test]
fn verifier_rejects_tampered_certificates() {
    let mut lp = LinearProgram::new(qv(&[-1, -1]));
    lp.a_ub = vec![qv(&[1, 2]), qv(&[3, 1])];
    lp.b_ub = qv(&[4, 6]);
    let c = optimal(&lp);
    let mut bad = c.clone();
    bad.y_ub[0] += r(1, 1000);
    assert!(bad.verify(&lp).is_err());
    let mut bad = c.clone();
    bad.x = qv(&[0, 0]);
    bad.objective = q(0);
    assert!(bad.verify(&lp).is_err());
}

/// A tiny deterministic generator so the random tests are reproducible.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
}

fn random_lp(g: &mut Lcg) -> LinearProgram {
    let n = g.int(1, 5) as usize;
    let m_ub = g.int(0, 5) as usize;
    let m_eq = g.int(0, 2) as usize;
    let mut lp = LinearProgram::new((0..n).map(|_| q(g.int(-5, 5))).collect());
    for _ in 0..m_ub {
        lp.a_ub.push((0..n).map(|_| q(g.int(-4, 4))).collect());
        lp.b_ub.push(q(g.int(-3, 10)));
    }
    for _ in 0..m_eq {
        lp.a_eq.push((0..n).map(|_| q(g.int(-3, 3))).collect());
        lp.b_eq.push(q(g.int(-4, 4)));
    }
    for b in lp.bounds.iter_mut() {
        *b = match g.int(0, 4) {
            0 => VarBound::free(),
            1 => VarBound::new(Some(q(g.int(-3, 0))), Some(q(g.int(1, 4)))),
            2 => VarBound::new(None, Some(q(g.int(-2, 3)))),
            _ => VarBound::nonnegative(),
        };
    }
    if g.int(0, 1) == 1 {
        lp.sense = Sense::Maximize;
    }
    lp
}

#[test]
fn random_programmes_all_certify() {
    let mut g = Lcg(0x5eed);
    let mut seen = [0usize; 3];
    for _ in 0..400 {
        let lp = random_lp(&mut g);
        let out = linprog(&lp).unwrap();
        out.verify(&lp).unwrap();
        seen[match out {
            LpOutcome::Optimal(_) => 0,
            LpOutcome::Infeasible(_) => 1,
            LpOutcome::Unbounded(_) => 2,
        }] += 1;
    }
    assert!(
        seen.iter().all(|&k| k > 10),
        "status mix too narrow: {seen:?}"
    );
}

#[test]
fn budget_trip_is_reported() {
    let mut lp = LinearProgram::new(vec![r(-3, 4), q(20), r(-1, 2), q(6)]);
    lp.a_ub = vec![
        vec![r(1, 4), q(-8), q(-1), q(9)],
        vec![r(1, 2), q(-12), r(-1, 2), q(3)],
        vec![q(0), q(0), q(1), q(0)],
    ];
    lp.b_ub = qv(&[0, 0, 1]);
    let _g = budget::enter(Budget {
        max_steps: Some(2),
        ..Budget::new()
    });
    let e = linprog(&lp).unwrap_err();
    assert!(matches!(
        e,
        LinprogError::Budget {
            source: BudgetError::Steps { .. },
            ..
        }
    ));
    assert_eq!(e.code(), "E-BUDGET-002");
}

// ---------------------------------------------------------------- MILP

fn brute_force(lp: &LinearProgram, lo: i64, hi: i64) -> Option<Rational> {
    let n = lp.nvars();
    let mut best: Option<Rational> = None;
    let mut x = vec![lo; n];
    loop {
        let xr: Vec<Rational> = x.iter().map(|&v| q(v)).collect();
        if lp.is_feasible(&xr) {
            let v = lp.objective_at(&xr);
            let better = match (&best, lp.sense) {
                (None, _) => true,
                (Some(b), Sense::Minimize) => v < *b,
                (Some(b), Sense::Maximize) => v > *b,
            };
            if better {
                best = Some(v);
            }
        }
        let mut k = 0;
        loop {
            if k == n {
                return best;
            }
            x[k] += 1;
            if x[k] <= hi {
                break;
            }
            x[k] = lo;
            k += 1;
        }
    }
}

#[test]
fn knapsack() {
    // max 10a + 13b + 7c + 8d  s.t. 4a + 6b + 3c + 5d ≤ 12, binary.
    let mut lp = LinearProgram::new(qv(&[10, 13, 7, 8]));
    lp.sense = Sense::Maximize;
    lp.a_ub = vec![qv(&[4, 6, 3, 5])];
    lp.b_ub = qv(&[12]);
    lp.bounds = vec![VarBound::new(Some(q(0)), Some(q(1))); 4];
    let out = milp(&lp, &MilpOptions::new(vec![true; 4])).unwrap();
    let MilpOutcome::Optimal(s) = out else {
        panic!("{out:?}")
    };
    assert_eq!(Some(s.objective.clone()), brute_force(&lp, 0, 1));
    assert_eq!(s.objective, q(25));
}

#[test]
fn assignment_problem() {
    // 3×3 assignment; optimum 1 + 2 + 2 = 5, checked by brute force.
    let cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]];
    let mut c = Vec::new();
    for row in cost {
        for v in row {
            c.push(q(v));
        }
    }
    let mut lp = LinearProgram::new(c);
    lp.bounds = vec![VarBound::new(Some(q(0)), Some(q(1))); 9];
    for i in 0..3 {
        let mut row = vec![q(0); 9];
        let mut col = vec![q(0); 9];
        for j in 0..3 {
            row[3 * i + j] = q(1);
            col[3 * j + i] = q(1);
        }
        lp.a_eq.push(row);
        lp.b_eq.push(q(1));
        lp.a_eq.push(col);
        lp.b_eq.push(q(1));
    }
    let MilpOutcome::Optimal(s) = milp(&lp, &MilpOptions::new(vec![true; 9])).unwrap() else {
        panic!()
    };
    assert_eq!(Some(s.objective.clone()), brute_force(&lp, 0, 1));
    assert_eq!(s.objective, q(5));
}

#[test]
fn integer_gap_differs_from_relaxation() {
    // max y  s.t. −x + y ≤ 1, 3x + 2y ≤ 12, 2x + 3y ≤ 12, x, y ≥ 0 integer.
    let mut lp = LinearProgram::new(qv(&[0, 1]));
    lp.sense = Sense::Maximize;
    lp.a_ub = vec![qv(&[-1, 1]), qv(&[3, 2]), qv(&[2, 3])];
    lp.b_ub = qv(&[1, 12, 12]);
    let relax = optimal(&lp).objective;
    assert_eq!(relax, r(14, 5)); // at (9/5, 14/5)
    let MilpOutcome::Optimal(s) = milp(&lp, &MilpOptions::new(vec![true; 2])).unwrap() else {
        panic!()
    };
    assert_eq!(s.objective, q(2));
}

#[test]
fn milp_infeasible_and_mixed() {
    // 2x = 1 with x integer: LP-feasible, integer-infeasible.
    let mut lp = LinearProgram::new(qv(&[1]));
    lp.a_eq = vec![qv(&[2])];
    lp.b_eq = qv(&[1]);
    assert!(matches!(
        milp(&lp, &MilpOptions::new(vec![true])).unwrap(),
        MilpOutcome::Infeasible { .. }
    ));
    // Mixed: x continuous, y integer: 2x + 2y = 3, min y ⇒ y = 0, x = 3/2.
    let mut lp = LinearProgram::new(qv(&[0, 1]));
    lp.a_eq = vec![qv(&[2, 2])];
    lp.b_eq = qv(&[3]);
    let MilpOutcome::Optimal(s) = milp(&lp, &MilpOptions::new(vec![false, true])).unwrap() else {
        panic!()
    };
    assert_eq!(s.x, vec![r(3, 2), q(0)]);
}

#[test]
fn milp_unbounded_has_integral_ray() {
    // min −x  s.t. x − 2y ≤ 1/2, x, y ≥ 0 integer: ray (2, 1) from (0, 0).
    let mut lp = LinearProgram::new(qv(&[-1, 0]));
    lp.a_ub = vec![vec![q(1), q(-2)]];
    lp.b_ub = vec![r(1, 2)];
    match milp(&lp, &MilpOptions::new(vec![true, true])).unwrap() {
        MilpOutcome::Unbounded(u) => {
            u.verify(&lp).unwrap();
            assert!(u.direction.iter().all(|v| *v.denom() == 1));
            assert!(u.point.iter().all(|v| *v.denom() == 1));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn milp_node_limit_reports_incumbent() {
    // A knapsack big enough to need more than 2 nodes.
    let mut lp = LinearProgram::new(qv(&[10, 13, 7, 8, 9, 11]));
    lp.sense = Sense::Maximize;
    lp.a_ub = vec![qv(&[4, 6, 3, 5, 4, 5])];
    lp.b_ub = qv(&[13]);
    lp.bounds = vec![VarBound::new(Some(q(0)), Some(q(1))); 6];
    let mut opts = MilpOptions::new(vec![true; 6]);
    opts.max_nodes = 3;
    opts.cut_rounds = 0;
    match milp(&lp, &opts).unwrap_err() {
        e @ LinprogError::NodeLimit { .. } => {
            assert_eq!(e.code(), "E-LP-005");
            if let LinprogError::NodeLimit {
                incumbent, bound, ..
            } = e
            {
                if let (Some(inc), Some(b)) = (incumbent, bound) {
                    assert!(lp.is_feasible(&inc.x));
                    assert!(b >= inc.objective);
                }
            }
        }
        other => panic!("{other:?}"),
    }
    let full = milp(&lp, &MilpOptions::new(vec![true; 6])).unwrap();
    let MilpOutcome::Optimal(s) = full else {
        panic!()
    };
    assert_eq!(Some(s.objective), brute_force(&lp, 0, 1));
}

#[test]
fn random_milps_match_brute_force() {
    let mut g = Lcg(0xb0b);
    for _ in 0..120 {
        let n = g.int(1, 3) as usize;
        let mut lp = LinearProgram::new((0..n).map(|_| q(g.int(-6, 6))).collect());
        for _ in 0..g.int(1, 4) {
            lp.a_ub.push((0..n).map(|_| q(g.int(-4, 4))).collect());
            lp.b_ub.push(q(g.int(-2, 9)));
        }
        lp.bounds = vec![VarBound::new(Some(q(-3)), Some(q(3))); n];
        if g.int(0, 1) == 1 {
            lp.sense = Sense::Maximize;
        }
        let expect = brute_force(&lp, -3, 3);
        match milp(&lp, &MilpOptions::new(vec![true; n])).unwrap() {
            MilpOutcome::Optimal(s) => assert_eq!(Some(s.objective), expect),
            MilpOutcome::Infeasible { .. } => assert_eq!(expect, None),
            MilpOutcome::Unbounded(_) => panic!("bounded box reported unbounded"),
        }
    }
}

// ---------------------------------------------------------------- exprs

#[test]
fn program_from_expressions() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let zero = pool.integer(0);
    // max 3x + 2y s.t. x + y ≤ 4, x + 3y ≤ 6, x ≥ 0, y ≥ 0, x ≤ 3.
    let obj = pool.add(vec![
        pool.mul(vec![pool.integer(3), x]),
        pool.mul(vec![pool.integer(2), y]),
    ]);
    let cons = vec![
        pool.pred_le(pool.add(vec![x, y]), pool.integer(4)),
        pool.pred_le(
            pool.add(vec![x, pool.mul(vec![pool.integer(3), y])]),
            pool.integer(6),
        ),
        pool.pred_and(vec![pool.pred_ge(x, zero), pool.pred_ge(y, zero)]),
        pool.pred_le(x, pool.integer(3)),
    ];
    let lp = program_from_exprs(&pool, obj, &cons, &[x, y], Sense::Maximize).unwrap();
    assert_eq!(lp.a_ub.len(), 5);
    let c = optimal(&lp);
    assert_eq!(c.objective, q(11));
    assert_eq!(c.x, qv(&[3, 1]));

    // Rational coefficients: x/2 ≤ 1.
    let half_x = pool.mul(vec![x, pool.rational(1, 2)]);
    let (coef, k) = linear_form(&pool, pool.add(vec![half_x, pool.integer(1)]), &[x, y]).unwrap();
    assert_eq!(coef, vec![r(1, 2), q(0)]);
    assert_eq!(k, q(1));

    // Refusals.
    let xy = pool.mul(vec![x, y]);
    assert_eq!(
        linear_form(&pool, xy, &[x, y]).unwrap_err().code(),
        "E-LP-003"
    );
    let p = pool.symbol("p", Domain::Real);
    assert_eq!(
        linear_form(&pool, pool.mul(vec![p, x]), &[x])
            .unwrap_err()
            .code(),
        "E-LP-003"
    );
    let sqrt2 = pool.pow(pool.integer(2), pool.rational(1, 2));
    assert_eq!(
        linear_form(&pool, pool.mul(vec![sqrt2, x]), &[x])
            .unwrap_err()
            .code(),
        "E-LP-002"
    );
    let strict = pool.pred_lt(x, pool.integer(1));
    assert_eq!(
        program_from_exprs(&pool, x, &[strict], &[x], Sense::Minimize)
            .unwrap_err()
            .code(),
        "E-LP-004"
    );
}
