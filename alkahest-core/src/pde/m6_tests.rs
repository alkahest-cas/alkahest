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

fn show_red(e: &Eq, sols: &[super::nonlinear::ReducedSolution]) {
    for r in sols {
        eprintln!(
            "  [{}] u = {}   consts={:?} assume={:?}",
            r.method,
            e.pool.display(r.value),
            r.constants
                .iter()
                .map(|&c| e.pool.display(c).to_string())
                .collect::<Vec<_>>(),
            r.assumptions
        );
    }
}

#[test]
fn traveling_waves() {
    for (src, want) in [
        ("u_t + u*u_x - u_xx", "burgers"),
        ("u_t + 6*u*u_x + u_xxx", "kdv"),
        ("u_t - u_xx - u + u^2", "fisher"),
        ("u_xt - sin(u)", "sine-gordon"),
        ("u_tt - u_xx - u + u^3", "phi4"),
        ("u_t + 6*u^2*u_x + u_xxx", "mkdv"),
    ] {
        let e = eq_of(src, &["x", "t"]);
        eprintln!("== {want}: {src}");
        match super::nonlinear::traveling_wave_solutions(e.eq, e.u, &e.pool) {
            Ok(s) => show_red(&e, &s),
            Err(err) => eprintln!("  error {err}"),
        }
    }
}

#[test]
fn similarity_heat_and_porous_medium() {
    for src in ["u_t - u_xx", "u_t - u*u_xx - u_x^2", "u_t + u*u_x - u_xx"] {
        let e = eq_of(src, &["x", "t"]);
        eprintln!("== {src}");
        match super::nonlinear::similarity_solutions(e.eq, e.u, &e.pool) {
            Ok(s) => show_red(&e, &s),
            Err(err) => eprintln!("  error {err}"),
        }
    }
}

#[test]
fn cole_hopf_verified_and_refused() {
    let e = eq_of("u_t + 2*u*u_x - 3*u_xx", &["x", "t"]);
    let ch = super::linearize::cole_hopf(e.eq, e.u, &e.pool).unwrap();
    assert!(ch.verified);
    eprintln!(
        "u = {}; {}",
        e.pool.display(ch.substitution),
        e.pool.display(ch.heat_equation)
    );
    let k = eq_of("u_t + u*u_x + u_xxx", &["x", "t"]);
    let err = super::linearize::cole_hopf(k.eq, k.u, &k.pool).unwrap_err();
    assert_eq!(crate::errors::AlkahestError::code(&err), "E-PDE-304");
}

#[test]
fn miura_mkdv_to_kdv() {
    let e = eq_of("u_t - 6*u^2*u_x + u_xxx", &["x", "t"]);
    let w = e.pool.func("w", e.vars.clone());
    let m = super::linearize::miura(e.eq, e.u, w, &e.pool).unwrap();
    eprintln!(
        "u = {}  kdv: {}",
        e.pool.display(m.substitution),
        e.pool.display(m.kdv)
    );
    assert!(m.verified);
}

#[test]
fn hodograph_shallow_water() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let t = pool.symbol("t", Domain::Real);
    let u = pool.func("u", vec![x, t]);
    let h = pool.func("h", vec![x, t]);
    let space = JetSpace::new(&[x, t], &[u, h], 2, &pool).unwrap();
    let mut syms: HashMap<String, ExprId> = HashMap::from([("x".into(), x), ("t".into(), t)]);
    let e1 = crate::parse::parse("u_t + u*u_x + h_x", &pool, &mut syms).unwrap();
    let e2 = crate::parse::parse("h_t + h*u_x + u*h_x", &pool, &mut syms).unwrap();
    let eqs = [space.from_jets(e1, &pool), space.from_jets(e2, &pool)];
    let hd = super::linearize::hodograph(&eqs, &[u, h], &pool).unwrap();
    assert!(hd.verified);
    for q in &hd.equations {
        eprintln!("hodograph: {}", pool.display(*q));
    }
}

#[test]
fn hopf_lax_quadratic() {
    let e = eq_of("u_t + u_x^2/2", &["x", "t"]);
    let hl = super::linearize::hopf_lax(e.eq, e.u, &e.pool).unwrap();
    assert!(hl.verified);
    eprintln!(
        "L = {}  objective = {}",
        e.pool.display(hl.lagrangian),
        e.pool.display(hl.objective)
    );
}

#[test]
fn pdsolve_m6_tiers() {
    use super::solve::{pdsolve, pdsolve_all, PdsolveOptions};
    for src in [
        "u_t + u*u_x - u_xx",
        "u_t + 6*u*u_x + u_xxx",
        "u_xt - sin(u)",
        "u_t - u_xx - u + u^2",
    ] {
        let e = eq_of(src, &["x", "t"]);
        let prob = super::problem::PdeProblem::new(&[e.eq], &[e.u], &e.pool).unwrap();
        let all = pdsolve_all(&prob, &PdsolveOptions::default(), &e.pool).unwrap();
        for s in &all {
            assert!(s.verification.status.is_verified());
            eprintln!("{src}: [{}] u = {}", s.method, e.pool.display(s.value));
        }
        let one = pdsolve(&prob, &PdsolveOptions::default(), &e.pool).unwrap();
        assert_eq!(one.kind, super::solve::PdeSolutionKind::Special);
    }
    // A linear second-order equation is still outside every tier.
    let h = eq_of("u_t - u_xx", &["x", "t"]);
    let prob = super::problem::PdeProblem::new(&[h.eq], &[h.u], &h.pool).unwrap();
    let err = pdsolve(&prob, &PdsolveOptions::default(), &h.pool).unwrap_err();
    assert_eq!(crate::errors::AlkahestError::code(&err), "E-PDE-050");
}
