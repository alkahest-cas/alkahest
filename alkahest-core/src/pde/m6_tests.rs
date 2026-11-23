//! Tests of the M6 layer: point symmetries, conservation laws, reductions.

use std::collections::HashMap;

use crate::kernel::{Domain, ExprId, ExprPool};
use crate::simplify::zero::ZeroStatus;

use super::jet::JetSpace;
use super::symmetry::*;

pub(crate) struct Eq {
    pub pool: ExprPool,
    pub vars: Vec<ExprId>,
    pub u: ExprId,
    pub eq: ExprId,
}

/// `src` written in jets (`u_t - u_xx`) over the variables `vars`.
pub(crate) fn eq_of(src: &str, vars: &[&str]) -> Eq {
    let pool = ExprPool::new();
    let xs: Vec<ExprId> = vars.iter().map(|v| pool.symbol(*v, Domain::Real)).collect();
    let u = pool.func("u", xs.clone());
    let space = JetSpace::new(&xs, &[u], 8, &pool).unwrap();
    let mut syms: HashMap<String, ExprId> = vars
        .iter()
        .zip(&xs)
        .map(|(v, &x)| (v.to_string(), x))
        .collect();
    let e = crate::parse::parse(src, &pool, &mut syms).unwrap();
    let eq = space.from_jets(e, &pool);
    Eq {
        pool,
        vars: xs,
        u,
        eq,
    }
}

fn syms(e: &Eq) -> Vec<PointSymmetries> {
    point_symmetries(e.eq, e.u, &e.pool).unwrap()
}

fn show(e: &Eq, cases: &[PointSymmetries]) {
    for c in cases {
        eprintln!(
            "case {:?} complete={} n={} notes={:?}",
            c.case
                .iter()
                .map(|(p, v)| format!("{} = {}", e.pool.display(*p), e.pool.display(*v)))
                .collect::<Vec<_>>(),
            c.complete,
            c.generators.len(),
            c.notes
        );
        for g in &c.generators {
            eprintln!(
                "   {}  verified={}",
                generator_display(g, &c.independent, c.dependent, &e.pool),
                g.verified
            );
        }
        if let Some(inf) = &c.infinite {
            eprintln!(
                "   infinite[{}]: {:?} with {:?}",
                inf.kind,
                inf.components
                    .iter()
                    .map(|&x| e.pool.display(x).to_string())
                    .collect::<Vec<_>>(),
                inf.constraints
                    .iter()
                    .map(|&x| e.pool.display(x).to_string())
                    .collect::<Vec<_>>()
            );
        }
    }
}

fn generic(cases: &[PointSymmetries]) -> &PointSymmetries {
    cases.iter().find(|c| c.case.is_empty()).unwrap()
}

#[test]
fn heat_equation_six_plus_superposition() {
    let e = eq_of("u_t - u_xx", &["x", "t"]);
    let cases = syms(&e);
    show(&e, &cases);
    let c = generic(&cases);
    assert_eq!(c.generators.len(), 6);
    assert!(c.generators.iter().all(|g| g.verified));
    assert!(c.complete);
    assert_eq!(c.infinite.as_ref().unwrap().kind, "superposition");
    assert!(c.structure_constants.is_some());
}

#[test]
fn burgers_five() {
    let e = eq_of("u_t + u*u_x - u_xx", &["x", "t"]);
    let cases = syms(&e);
    show(&e, &cases);
    let c = generic(&cases);
    assert_eq!(c.generators.len(), 5);
    assert!(c.complete && c.infinite.is_none());
}

#[test]
fn kdv_four() {
    let e = eq_of("u_t + u*u_x + u_xxx", &["x", "t"]);
    let cases = syms(&e);
    show(&e, &cases);
    let c = generic(&cases);
    assert_eq!(c.generators.len(), 4);
    assert!(c.complete);
}

#[test]
fn sine_gordon_three() {
    let e = eq_of("u_xt - sin(u)", &["x", "t"]);
    let cases = syms(&e);
    show(&e, &cases);
    let c = generic(&cases);
    assert_eq!(c.generators.len(), 3);
    assert!(c.complete);
}

#[test]
fn verify_symmetry_refutes_non_symmetry() {
    let e = eq_of("u_t - u_xx", &["x", "t"]);
    let p = &e.pool;
    let u = p.symbol("u", Domain::Real);
    let (x, t) = (e.vars[0], e.vars[1]);
    // x ∂_x + 2t ∂_t is a symmetry; x ∂_x + t ∂_t is not.
    let good = [x, p.mul(vec![p.integer(2), t]), p.integer(0)];
    let bad = [x, t, p.integer(0)];
    assert_eq!(
        verify_symmetry(e.eq, e.u, &good, p).unwrap(),
        ZeroStatus::ProvedZero
    );
    assert_ne!(
        verify_symmetry(e.eq, e.u, &bad, p).unwrap(),
        ZeroStatus::ProvedZero
    );
    let _ = u;
}

#[test]
fn wave_and_laplace_infinite() {
    for src in ["u_tt - u_xx", "u_xx + u_tt"] {
        let e = eq_of(src, &["x", "t"]);
        let cases = syms(&e);
        show(&e, &cases);
        let c = generic(&cases);
        assert!(c.infinite.is_some());
        assert!(c.generators.iter().all(|g| g.verified));
    }
}

#[test]
fn nonlinear_diffusion_classification() {
    let e = eq_of("u_t - u^m*u_xx - m*u^(m-1)*u_x^2", &["x", "t"]);
    let cases = syms(&e);
    show(&e, &cases);
    let c = generic(&cases);
    assert_eq!(c.generators.len(), 4);
}

#[test]
fn exponential_diffusion_and_fisher() {
    let e = eq_of("u_t - exp(u)*u_xx - exp(u)*u_x^2", &["x", "t"]);
    let cases = syms(&e);
    show(&e, &cases);
    assert_eq!(generic(&cases).generators.len(), 4);
    let e = eq_of("u_t - u_xx - u + u^2", &["x", "t"]);
    let cases = syms(&e);
    show(&e, &cases);
    assert_eq!(generic(&cases).generators.len(), 2);
}

#[test]
fn kdv_conservation_laws() {
    let e = eq_of("u_t + u*u_x + u_xxx", &["x", "t"]);
    let ms = super::conservation::conservation_multipliers(e.eq, e.u, 2, &e.pool).unwrap();
    for m in &ms {
        eprintln!(
            "multipliers {:?} complete={} notes={:?}",
            m.multipliers
                .iter()
                .map(|&x| e.pool.display(x).to_string())
                .collect::<Vec<_>>(),
            m.complete,
            m.notes
        );
    }
    let laws = super::conservation::conservation_laws(e.eq, e.u, 2, &e.pool).unwrap();
    for l in &laws {
        eprintln!(
            "  Λ = {}  Φ = {:?} verified={}",
            e.pool.display(l.multiplier),
            l.components
                .iter()
                .map(|&x| e.pool.display(x).to_string())
                .collect::<Vec<_>>(),
            l.verified
        );
    }
    assert_eq!(laws.len(), 4);
    assert!(laws.iter().all(|l| l.verified));
}
