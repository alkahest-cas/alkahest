//! Tests for the revised simplex and warm starts, Gomory cuts in `milp`, and
//! parametric LP.

use rug::Rational;

use super::*;
use crate::budget::{self, Budget};
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprId, ExprPool};

fn q(n: i64) -> Rational {
    Rational::from(n)
}

fn r(n: i64, d: i64) -> Rational {
    Rational::from((n, d))
}

fn qv(v: &[i64]) -> Vec<Rational> {
    v.iter().map(|&n| q(n)).collect()
}

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

fn random_lp(g: &mut Lcg, nmax: i64, mmax: i64) -> LinearProgram {
    let n = g.int(1, nmax) as usize;
    let m_ub = g.int(0, mmax) as usize;
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

/// A random sparse programme with `m` rows and `n` columns, feasible at
/// `x = 1` and bounded by `x ≤ 10`.
fn sparse_lp(g: &mut Lcg, m: usize, n: usize, per_row: usize) -> LinearProgram {
    let mut lp = LinearProgram::new((0..n).map(|_| q(g.int(-9, 9))).collect());
    lp.sense = Sense::Maximize;
    for _ in 0..m {
        let mut row = vec![q(0); n];
        for _ in 0..per_row {
            let j = g.int(0, n as i64 - 1) as usize;
            row[j] = q(g.int(1, 9));
        }
        let s: Rational = row.iter().fold(q(0), |a, b| a + b);
        lp.b_ub.push(s + q(g.int(0, 20)));
        lp.a_ub.push(row);
    }
    lp.bounds = vec![VarBound::new(Some(q(0)), Some(q(10))); n];
    lp
}

fn status(o: &LpOutcome) -> u8 {
    match o {
        LpOutcome::Optimal(_) => 0,
        LpOutcome::Infeasible(_) => 1,
        LpOutcome::Unbounded(_) => 2,
    }
}

fn objective(o: &LpOutcome) -> Option<Rational> {
    match o {
        LpOutcome::Optimal(c) => Some(c.objective.clone()),
        _ => None,
    }
}

fn with(method: LpMethod) -> LpOptions {
    LpOptions::default().with_method(method)
}

// ---------------------------------------------------------- revised simplex

#[test]
fn revised_agrees_with_dense_on_random_programmes() {
    let mut g = Lcg(0xf19);
    let mut seen = [0usize; 3];
    for _ in 0..500 {
        let lp = random_lp(&mut g, 6, 7);
        let d = linprog_with(&lp, &with(LpMethod::Dense)).unwrap();
        let rv = linprog_with(&lp, &with(LpMethod::Revised)).unwrap();
        assert_eq!(d.method, LpMethod::Dense);
        assert_eq!(rv.method, LpMethod::Revised);
        rv.outcome.verify(&lp).unwrap();
        assert_eq!(status(&d.outcome), status(&rv.outcome), "{lp:?}");
        assert_eq!(objective(&d.outcome), objective(&rv.outcome), "{lp:?}");
        seen[status(&rv.outcome) as usize] += 1;
    }
    assert!(seen.iter().all(|&k| k > 20), "{seen:?}");
}

#[test]
fn revised_terminates_on_cycling_examples() {
    // Beale's example (cycles under the textbook largest-coefficient rule).
    let mut lp = LinearProgram::new(vec![r(-3, 4), q(20), r(-1, 2), q(6)]);
    lp.a_ub = vec![
        vec![r(1, 4), q(-8), q(-1), q(9)],
        vec![r(1, 2), q(-12), r(-1, 2), q(3)],
        vec![q(0), q(0), q(1), q(0)],
    ];
    lp.b_ub = qv(&[0, 0, 1]);
    let s = linprog_with(&lp, &with(LpMethod::Revised)).unwrap();
    assert_eq!(objective(&s.outcome), Some(r(-5, 4)));
}

#[test]
fn revised_agrees_on_medium_sparse_programmes() {
    let mut g = Lcg(0x5a5a);
    for &(m, n) in &[(30usize, 40usize), (60, 50)] {
        let lp = sparse_lp(&mut g, m, n, 4);
        let d = linprog_with(&lp, &with(LpMethod::Dense)).unwrap();
        let rv = linprog_with(&lp, &with(LpMethod::Revised)).unwrap();
        assert_eq!(objective(&d.outcome), objective(&rv.outcome));
        assert!(objective(&rv.outcome).is_some());
    }
}

#[test]
fn auto_picks_dense_for_small_and_revised_for_large() {
    let mut lp = LinearProgram::new(qv(&[1, 1]));
    lp.a_ub = vec![qv(&[1, 2])];
    lp.b_ub = qv(&[4]);
    assert_eq!(
        linprog_with(&lp, &LpOptions::default()).unwrap().method,
        LpMethod::Dense
    );
    let mut g = Lcg(7);
    let big = sparse_lp(&mut g, 20, 20, 3);
    assert_eq!(
        linprog_with(&big, &LpOptions::default()).unwrap().method,
        LpMethod::Revised
    );
}

#[test]
fn revised_budget_trip_is_reported() {
    let mut g = Lcg(11);
    let lp = sparse_lp(&mut g, 30, 30, 4);
    let _g = budget::enter(Budget {
        max_steps: Some(3),
        ..Budget::new()
    });
    let e = linprog_with(&lp, &with(LpMethod::Revised)).unwrap_err();
    assert!(matches!(e, LinprogError::Budget { .. }), "{e:?}");
}

/// Timing comparison, run by hand:
/// `cargo test -p alkahest-cas --release --lib bench_revised_vs_dense -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_revised_vs_dense() {
    let mut g = Lcg(0xbe7c);
    let sizes = [
        (25usize, 25usize, false),
        (50, 50, false),
        (100, 100, false),
        (200, 200, false),
        (300, 300, false),
        (500, 500, false),
        (25, 25, true),
        (50, 50, true),
        (100, 100, true),
        (200, 200, true),
        (300, 300, true),
    ];
    for &(m, n, cover) in &sizes {
        let mut lp = sparse_lp(&mut g, m, n, 5);
        if cover {
            // Every other row becomes a covering row `a·x ≥ (a·1)/2`, which
            // needs phase 1.
            for i in (0..m).step_by(2) {
                let s: Rational = lp.a_ub[i].iter().fold(q(0), |a, b| a + b);
                lp.a_ub[i] = lp.a_ub[i].iter().map(|v| -v.clone()).collect();
                lp.b_ub[i] = -(s / q(2));
            }
        }
        let t = std::time::Instant::now();
        let rv = linprog_with(&lp, &with(LpMethod::Revised)).unwrap();
        let tr = t.elapsed();
        let (td, dense_obj) = if m <= 100 {
            let t = std::time::Instant::now();
            let d = linprog_with(&lp, &with(LpMethod::Dense)).unwrap();
            (Some(t.elapsed()), objective(&d.outcome))
        } else {
            (None, None)
        };
        if td.is_some() {
            assert_eq!(dense_obj, objective(&rv.outcome));
        }
        println!(
            "m={m} n={n} covering={cover}: revised {:?} ({} pivots), dense {:?}",
            tr, rv.iterations, td
        );
    }
}

// --------------------------------------------------------------- warm start

#[test]
fn warm_start_after_rhs_change_uses_dual_simplex() {
    let mut g = Lcg(0xabc);
    let mut warmed = 0;
    for _ in 0..40 {
        let mut lp = sparse_lp(&mut g, 15, 12, 3);
        let first = linprog_with(&lp, &with(LpMethod::Revised)).unwrap();
        // Tighten a few right-hand sides: the old basis stays dual feasible.
        for i in 0..lp.b_ub.len() {
            if g.int(0, 2) == 0 {
                lp.b_ub[i] -= q(g.int(1, 15));
            }
        }
        let cold = linprog_with(&lp, &with(LpMethod::Revised)).unwrap();
        let warm =
            linprog_with(&lp, &LpOptions::default().with_basis(first.basis.clone())).unwrap();
        warm.outcome.verify(&lp).unwrap();
        assert_eq!(status(&cold.outcome), status(&warm.outcome));
        assert_eq!(objective(&cold.outcome), objective(&warm.outcome));
        if warm.warm_started {
            warmed += 1;
        }
    }
    assert!(warmed >= 35, "only {warmed} warm starts were usable");
}

#[test]
fn warm_start_detects_infeasibility_with_farkas() {
    // max x + y, x + y ≤ 4, x ≤ 3, y ≤ 3, x, y ≥ 0; then demand x + y ≥ 5.
    let mut lp = LinearProgram::new(qv(&[1, 1]));
    lp.sense = Sense::Maximize;
    lp.a_ub = vec![qv(&[1, 1]), qv(&[1, 0]), qv(&[0, 1])];
    lp.b_ub = qv(&[4, 3, 3]);
    let first = linprog_with(&lp, &with(LpMethod::Revised)).unwrap();
    lp.a_ub.push(qv(&[-1, -1]));
    lp.b_ub.push(q(-5));
    let warm = linprog_with(&lp, &LpOptions::default().with_basis(first.basis)).unwrap();
    assert!(warm.warm_started);
    match warm.outcome {
        LpOutcome::Infeasible(f) => f.verify(&lp).unwrap(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn warm_start_after_objective_change_and_with_junk_basis() {
    let mut g = Lcg(0xdef);
    for _ in 0..30 {
        let mut lp = sparse_lp(&mut g, 12, 10, 3);
        let first = linprog_with(&lp, &with(LpMethod::Revised)).unwrap();
        for c in lp.c.iter_mut() {
            *c += q(g.int(-3, 3));
        }
        let warm =
            linprog_with(&lp, &LpOptions::default().with_basis(first.basis.clone())).unwrap();
        assert!(warm.warm_started);
        let dense = linprog_with(&lp, &with(LpMethod::Dense)).unwrap();
        assert_eq!(objective(&dense.outcome), objective(&warm.outcome));
    }
    // A basis naming columns that do not exist (or duplicates) is ignored
    // where it does not apply.
    let lp = sparse_lp(&mut g, 8, 6, 2);
    let junk = LpBasis::new(vec![
        BasisEntry::Var(99),
        BasisEntry::Slack(99),
        BasisEntry::VarNeg(0),
        BasisEntry::UpperSlack(3),
        BasisEntry::UpperSlack(3),
        BasisEntry::Var(0),
    ]);
    let s = linprog_with(&lp, &LpOptions::default().with_basis(junk)).unwrap();
    let d = linprog_with(&lp, &with(LpMethod::Dense)).unwrap();
    assert_eq!(objective(&s.outcome), objective(&d.outcome));
}

#[test]
fn warm_start_on_random_programmes_with_changes() {
    let mut g = Lcg(0x77);
    for _ in 0..300 {
        let mut lp = random_lp(&mut g, 5, 6);
        let first = linprog_with(&lp, &with(LpMethod::Revised)).unwrap();
        match g.int(0, 2) {
            0 => {
                for b in lp.b_ub.iter_mut().chain(lp.b_eq.iter_mut()) {
                    *b += q(g.int(-2, 2));
                }
            }
            1 => {
                for c in lp.c.iter_mut() {
                    *c += q(g.int(-2, 2));
                }
            }
            _ => {
                let n = lp.nvars();
                lp.a_ub.push((0..n).map(|_| q(g.int(-3, 3))).collect());
                lp.b_ub.push(q(g.int(-2, 6)));
            }
        }
        let warm = linprog_with(&lp, &LpOptions::default().with_basis(first.basis)).unwrap();
        let dense = linprog_with(&lp, &with(LpMethod::Dense)).unwrap();
        assert_eq!(status(&dense.outcome), status(&warm.outcome), "{lp:?}");
        assert_eq!(objective(&dense.outcome), objective(&warm.outcome));
    }
}

// -------------------------------------------------------------- Gomory cuts

/// Every integer point of the box `[lo, hi]ⁿ` feasible for `lp`.
fn integer_points(lp: &LinearProgram, lo: i64, hi: i64) -> Vec<Vec<Rational>> {
    let n = lp.nvars();
    let mut out = Vec::new();
    let mut x = vec![lo; n];
    loop {
        let xr: Vec<Rational> = x.iter().map(|&v| q(v)).collect();
        if lp.is_feasible(&xr) {
            out.push(xr);
        }
        let mut k = 0;
        loop {
            if k == n {
                return out;
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

fn best(lp: &LinearProgram, pts: &[Vec<Rational>]) -> Option<Rational> {
    pts.iter()
        .map(|x| lp.objective_at(x))
        .reduce(|a, b| match lp.sense {
            Sense::Minimize => a.min(b),
            Sense::Maximize => a.max(b),
        })
}

#[test]
fn gomory_cut_closes_a_simple_gap() {
    // max x + y  s.t. 2x + 2y ≤ 3, x, y ∈ ℤ≥0: LP 3/2, integer 1.
    let mut lp = LinearProgram::new(qv(&[1, 1]));
    lp.sense = Sense::Maximize;
    lp.a_ub = vec![qv(&[2, 2])];
    lp.b_ub = qv(&[3]);
    let rep = milp_with(
        &lp,
        &MilpOptions::new(vec![true; 2]),
        &CutOptions::default(),
    )
    .unwrap();
    let MilpOutcome::Optimal(s) = rep.outcome else {
        panic!()
    };
    assert_eq!(s.objective, q(1));
    assert!(!rep.cuts.is_empty());
    // The cut closes the root gap, so no branching is needed.
    assert_eq!(s.nodes, 1);
}

#[test]
fn gomory_cuts_never_remove_integer_points() {
    let mut g = Lcg(0x6060);
    let mut with_cuts = 0;
    for _ in 0..250 {
        let n = g.int(2, 3) as usize;
        let mut lp = LinearProgram::new((0..n).map(|_| q(g.int(-6, 6))).collect());
        for _ in 0..g.int(1, 4) {
            lp.a_ub
                .push((0..n).map(|_| r(g.int(-7, 7), g.int(1, 3))).collect());
            lp.b_ub.push(r(g.int(-2, 13), g.int(1, 2)));
        }
        if g.int(0, 2) == 0 {
            lp.a_eq.push((0..n).map(|_| q(g.int(-2, 2))).collect());
            lp.b_eq.push(q(g.int(-2, 2)));
        }
        lp.bounds = (0..n)
            .map(|_| match g.int(0, 2) {
                0 => VarBound::new(Some(r(-7, 2)), Some(r(10, 3))),
                1 => VarBound::new(Some(q(-3)), Some(q(3))),
                _ => VarBound::new(Some(r(-5, 2)), Some(q(3))),
            })
            .collect();
        if g.int(0, 1) == 1 {
            lp.sense = Sense::Maximize;
        }
        let pts = integer_points(&lp, -4, 4);
        let cut_opts = CutOptions {
            cut_rounds: 10,
            ..CutOptions::default()
        };
        let rep = milp_with(&lp, &MilpOptions::new(vec![true; n]), &cut_opts).unwrap();
        match rep.outcome {
            MilpOutcome::Optimal(s) => {
                assert_eq!(Some(s.objective.clone()), best(&lp, &pts), "{lp:?}");
                if !rep.cuts.is_empty() {
                    with_cuts += 1;
                }
                for (row, rhs) in &rep.cuts {
                    for x in &pts {
                        assert!(dot(row, x) <= *rhs, "cut {row:?} <= {rhs} removes {x:?}");
                    }
                }
            }
            MilpOutcome::Infeasible { .. } => assert!(pts.is_empty(), "{lp:?}"),
            MilpOutcome::Unbounded(_) => panic!("bounded box reported unbounded"),
        }
    }
    assert!(with_cuts > 40, "only {with_cuts} problems produced cuts");
}

#[test]
fn mixed_integer_cuts_keep_the_optimum() {
    let mut g = Lcg(0x3131);
    for _ in 0..120 {
        let n = g.int(2, 4) as usize;
        let mut lp = LinearProgram::new((0..n).map(|_| q(g.int(-6, 6))).collect());
        for _ in 0..g.int(1, 4) {
            lp.a_ub
                .push((0..n).map(|_| r(g.int(-7, 7), g.int(1, 2))).collect());
            lp.b_ub.push(r(g.int(-2, 13), g.int(1, 3)));
        }
        lp.bounds = (0..n)
            .map(|_| match g.int(0, 3) {
                0 => VarBound::free(),
                1 => VarBound::new(None, Some(r(7, 2))),
                _ => VarBound::new(Some(r(-5, 2)), Some(q(4))),
            })
            .collect();
        if g.int(0, 1) == 1 {
            lp.sense = Sense::Maximize;
        }
        let integrality: Vec<bool> = (0..n).map(|_| g.int(0, 2) > 0).collect();
        let mut opts = MilpOptions::new(integrality.clone());
        opts.max_nodes = 5000;
        let a = milp_with(&lp, &opts, &CutOptions::none()).map(|r| r.outcome);
        let rep = milp_with(&lp, &opts, &CutOptions::default());
        let cuts = rep.as_ref().map(|r| r.cuts.clone()).unwrap_or_default();
        let b = rep.map(|r| r.outcome);
        match (a, b) {
            (Ok(MilpOutcome::Optimal(a)), Ok(MilpOutcome::Optimal(b))) => {
                assert_eq!(a.objective, b.objective, "{lp:?} {integrality:?}");
                // Every cut keeps the plain search's optimum.
                for (row, rhs) in &cuts {
                    assert!(dot(row, &a.x) <= *rhs);
                }
            }
            (Ok(MilpOutcome::Infeasible { .. }), Ok(MilpOutcome::Infeasible { .. })) => {}
            (Ok(MilpOutcome::Unbounded(_)), Ok(MilpOutcome::Unbounded(_))) => {}
            (Err(_), _) | (_, Err(_)) => {}
            (a, b) => panic!("{a:?} vs {b:?} on {lp:?} {integrality:?}"),
        }
    }
}

#[test]
fn cut_limits_are_honoured() {
    let mut lp = LinearProgram::new(qv(&[1, 1, 1]));
    lp.sense = Sense::Maximize;
    lp.a_ub = vec![qv(&[2, 2, 0]), qv(&[0, 2, 2]), qv(&[2, 0, 2])];
    lp.b_ub = qv(&[3, 3, 3]);
    let opts = MilpOptions::new(vec![true; 3]);
    let mut cut_opts = CutOptions {
        max_cuts: 1,
        ..CutOptions::default()
    };
    let rep = milp_with(&lp, &opts, &cut_opts).unwrap();
    let MilpOutcome::Optimal(s) = rep.outcome else {
        panic!()
    };
    assert!(rep.cuts.len() <= 1);
    assert_eq!(s.objective, q(1));
    cut_opts.cut_rounds = 0;
    let rep = milp_with(&lp, &opts, &cut_opts).unwrap();
    let MilpOutcome::Optimal(s) = rep.outcome else {
        panic!()
    };
    assert!(rep.cuts.is_empty());
    assert_eq!(s.objective, q(1));
}

// ------------------------------------------------------------- parametric

fn check_parametric(lp: &LinearProgram, delta: &ParametricDelta, sol: &ParametricSolution) {
    // Sample every breakpoint, midpoints, and points beyond the ends.
    let mut samples: Vec<Rational> = Vec::new();
    for b in &sol.breakpoints {
        samples.push(b.clone());
        samples.push(Rational::from(b - &r(1, 7)));
        samples.push(Rational::from(b + &r(1, 7)));
    }
    for w in sol.breakpoints.windows(2) {
        samples.push(Rational::from(&w[0] + &w[1]) / 2);
    }
    for t in [-50, -7, -1, 0, 1, 3, 7, 50] {
        samples.push(q(t));
    }
    for t in samples {
        let Some(reg) = sol.region_at(&t) else {
            continue;
        };
        let lp_t = parametric::lp_at(lp, delta, &t);
        let out = linprog(&lp_t).unwrap();
        let expected = match &out {
            LpOutcome::Optimal(_) => RegionStatus::Optimal,
            LpOutcome::Infeasible(_) => RegionStatus::Infeasible,
            LpOutcome::Unbounded(_) => RegionStatus::Unbounded,
        };
        assert_eq!(reg.status, expected, "θ = {t}: {sol:?}");
        assert_eq!(sol.value_at(&t), objective(&out), "θ = {t}");
        if let Some(cert) = reg.certificate_at(&t) {
            cert.verify(&lp_t).unwrap();
        }
    }
    // Regions tile the range left to right.
    for w in sol.regions.windows(2) {
        assert_eq!(w[0].upper, w[1].lower);
    }
}

#[test]
fn parametric_rhs_textbook() {
    // max x + y  s.t.  x + 2y ≤ 4 + θ,  3x + y ≤ 6,  x, y ≥ 0.
    let mut lp = LinearProgram::new(qv(&[1, 1]));
    lp.sense = Sense::Maximize;
    lp.a_ub = vec![qv(&[1, 2]), qv(&[3, 1])];
    lp.b_ub = qv(&[4, 6]);
    let delta = ParametricDelta::Rhs {
        b_ub: qv(&[1, 0]),
        b_eq: vec![],
    };
    let sol = linprog_parametric(&lp, &delta, None, None).unwrap();
    // Infeasible below −4; 4 + θ on [−4, −2]; (14 + 2θ)/5 on [−2, 8]; 6 after.
    assert_eq!(sol.breakpoints, vec![q(-4), q(-2), q(8)]);
    let st: Vec<RegionStatus> = sol.regions.iter().map(|r| r.status).collect();
    assert_eq!(
        st,
        vec![
            RegionStatus::Infeasible,
            RegionStatus::Optimal,
            RegionStatus::Optimal,
            RegionStatus::Optimal
        ]
    );
    assert_eq!(sol.regions[1].value, Some((q(4), q(1))));
    assert_eq!(sol.regions[2].value, Some((r(14, 5), r(2, 5))));
    assert_eq!(sol.regions[3].value, Some((q(6), q(0))));
    check_parametric(&lp, &delta, &sol);
}

#[test]
fn parametric_objective_textbook() {
    // max x + θ·y  s.t.  x + 2y ≤ 4,  3x + y ≤ 6,  x, y ≥ 0.
    let mut lp = LinearProgram::new(qv(&[1, 0]));
    lp.sense = Sense::Maximize;
    lp.a_ub = vec![qv(&[1, 2]), qv(&[3, 1])];
    lp.b_ub = qv(&[4, 6]);
    let delta = ParametricDelta::Objective { c: qv(&[0, 1]) };
    let sol = linprog_parametric(&lp, &delta, None, None).unwrap();
    // Vertices (2,0), (8/5, 6/5), (0, 2): switches at θ = 1/3 and θ = 2.
    assert_eq!(sol.breakpoints, vec![r(1, 3), q(2)]);
    assert_eq!(sol.value_at(&q(0)), Some(q(2)));
    assert_eq!(sol.value_at(&q(1)), Some(r(14, 5)));
    assert_eq!(sol.value_at(&q(5)), Some(q(10)));
    check_parametric(&lp, &delta, &sol);
}

#[test]
fn parametric_unbounded_and_infeasible_flanks() {
    // min θ·x − y  s.t.  y ≤ x + 1, y ≤ 3 − x (free): bounded iff |θ| ≤ 1.
    let mut lp = LinearProgram::new(qv(&[0, -1]));
    lp.a_ub = vec![qv(&[-1, 1]), qv(&[1, 1])];
    lp.b_ub = qv(&[1, 3]);
    lp.bounds = vec![VarBound::free(), VarBound::free()];
    let delta = ParametricDelta::Objective { c: qv(&[1, 0]) };
    let sol = linprog_parametric(&lp, &delta, Some(q(-5)), Some(q(5))).unwrap();
    let st: Vec<RegionStatus> = sol.regions.iter().map(|r| r.status).collect();
    assert_eq!(st.first(), Some(&RegionStatus::Unbounded));
    assert_eq!(st.last(), Some(&RegionStatus::Unbounded));
    assert_eq!(sol.breakpoints, vec![q(-1), q(1)]);
    assert_eq!(sol.value_at(&r(1, 2)), Some(r(-3, 2)));
    check_parametric(&lp, &delta, &sol);

    // An RHS that becomes infeasible at both ends: 1 + θ ≤ x ≤ 3 − θ.
    let mut lp = LinearProgram::new(qv(&[1]));
    lp.a_ub = vec![qv(&[-1]), qv(&[1])];
    lp.b_ub = qv(&[-1, 3]);
    lp.bounds = vec![VarBound::free()];
    let delta = ParametricDelta::Rhs {
        b_ub: qv(&[-1, -1]),
        b_eq: vec![],
    };
    let sol = linprog_parametric(&lp, &delta, None, None).unwrap();
    assert_eq!(sol.breakpoints, vec![q(1)]);
    assert_eq!(sol.status_at(&q(2)), Some(RegionStatus::Infeasible));
    assert_eq!(sol.value_at(&q(0)), Some(q(1)));
    assert_eq!(sol.value_at(&q(1)), Some(q(2)));
    check_parametric(&lp, &delta, &sol);
}

#[test]
fn parametric_unbounded_wherever_feasible_owns_its_closed_end() {
    // min −y  s.t.  x ≤ θ − 1,  x ≥ 0,  y free: feasible iff θ ≥ 1, never bounded.
    let mut lp = LinearProgram::new(qv(&[0, -1]));
    lp.a_ub = vec![qv(&[1, 0])];
    lp.b_ub = qv(&[-1]);
    lp.bounds = vec![VarBound::nonnegative(), VarBound::free()];
    let delta = ParametricDelta::Rhs {
        b_ub: qv(&[1]),
        b_eq: vec![],
    };
    let sol = linprog_parametric(&lp, &delta, Some(q(-3)), None).unwrap();
    assert_eq!(sol.breakpoints, vec![q(1)]);
    assert_eq!(sol.status_at(&q(0)), Some(RegionStatus::Infeasible));
    assert_eq!(sol.status_at(&q(1)), Some(RegionStatus::Unbounded));
    check_parametric(&lp, &delta, &sol);
}

#[test]
fn parametric_random_programmes_match_pointwise_solves() {
    let mut g = Lcg(0x9a9a);
    let mut regions = 0;
    for k in 0..160 {
        let lp = random_lp(&mut g, 4, 5);
        let delta = if k % 2 == 0 {
            ParametricDelta::Rhs {
                b_ub: lp.b_ub.iter().map(|_| q(g.int(-3, 3))).collect(),
                b_eq: lp.b_eq.iter().map(|_| q(g.int(-2, 2))).collect(),
            }
        } else {
            ParametricDelta::Objective {
                c: lp.c.iter().map(|_| q(g.int(-3, 3))).collect(),
            }
        };
        let (lo, hi) = match g.int(0, 2) {
            0 => (None, None),
            1 => (Some(q(-6)), Some(q(6))),
            _ => (Some(q(g.int(-3, 0))), None),
        };
        let sol = linprog_parametric(&lp, &delta, lo, hi).unwrap();
        regions += sol.regions.len();
        check_parametric(&lp, &delta, &sol);
    }
    assert!(regions > 200, "{regions}");
}

#[test]
fn parametric_shape_errors() {
    let lp = LinearProgram::new(qv(&[1, 1]));
    let bad = ParametricDelta::Objective { c: qv(&[1]) };
    assert_eq!(
        linprog_parametric(&lp, &bad, None, None)
            .unwrap_err()
            .code(),
        "E-LP-001"
    );
    let ok = ParametricDelta::Objective { c: qv(&[1, 0]) };
    assert_eq!(
        linprog_parametric(&lp, &ok, Some(q(2)), Some(q(1)))
            .unwrap_err()
            .code(),
        "E-LP-001"
    );
}

#[test]
fn parametric_from_expressions_and_refusals() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let t = pool.symbol("t", Domain::Real);
    let zero = pool.integer(0);
    let nonneg = pool.pred_and(vec![pool.pred_ge(x, zero), pool.pred_ge(y, zero)]);
    // max x + y  s.t.  x + 2y ≤ 4 + t,  3x + y ≤ 6.
    let cons = vec![
        pool.pred_le(
            pool.add(vec![x, pool.mul(vec![pool.integer(2), y])]),
            pool.add(vec![pool.integer(4), t]),
        ),
        pool.pred_le(
            pool.add(vec![pool.mul(vec![pool.integer(3), x]), y]),
            pool.integer(6),
        ),
        nonneg,
    ];
    let obj = pool.add(vec![x, y]);
    let (lp, delta) =
        parametric_from_exprs(&pool, obj, &cons, &[x, y], t, Sense::Maximize).unwrap();
    assert_eq!(
        delta,
        ParametricDelta::Rhs {
            b_ub: qv(&[1, 0, 0, 0]),
            b_eq: vec![]
        }
    );
    let sol = linprog_parametric(&lp, &delta, None, None).unwrap();
    assert_eq!(sol.breakpoints, vec![q(-4), q(-2), q(8)]);

    // Objective (2 + t)·x + y.
    let obj2 = pool.add(vec![
        pool.mul(vec![pool.add(vec![pool.integer(2), t]), x]),
        y,
    ]);
    let (_, d2) =
        parametric_from_exprs(&pool, obj2, &cons[1..], &[x, y], t, Sense::Maximize).unwrap();
    assert_eq!(d2, ParametricDelta::Objective { c: qv(&[1, 0]) });

    // Refusals.
    let code = |o: ExprId, cs: &[ExprId]| {
        parametric_from_exprs(&pool, o, cs, &[x, y], t, Sense::Minimize)
            .unwrap_err()
            .code()
    };
    let in_matrix = pool.pred_le(pool.mul(vec![t, x]), pool.integer(1));
    assert_eq!(code(obj, &[in_matrix]), "E-LP-007");
    assert_eq!(code(obj2, &cons), "E-LP-007");
    let t_sq = pool.mul(vec![pool.pow(t, pool.integer(2)), x]);
    assert_eq!(code(t_sq, &[]), "E-LP-007");
    let sin_t = pool.pred_le(x, pool.func("sin", vec![t]));
    assert_eq!(code(obj, &[sin_t]), "E-LP-007");
    let xy = pool.mul(vec![x, y]);
    assert_eq!(code(xy, &[]), "E-LP-003");
}
