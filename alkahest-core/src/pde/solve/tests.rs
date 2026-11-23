use super::*;
use crate::errors::AlkahestError;
use crate::kernel::Domain;

struct Fx {
    pool: ExprPool,
    x: ExprId,
    y: ExprId,
    u: ExprId,
}

fn fx() -> Fx {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let u = pool.func("u", vec![x, y]);
    Fx { pool, x, y, u }
}

fn dd(f: &Fx, v: ExprId) -> ExprId {
    crate::diff::diff_impl::diff(f.u, v, &f.pool).unwrap().value
}

fn solve(f: &Fx, eq: ExprId) -> Result<PdeSolution, PdeError> {
    let prob = PdeProblem::new(&[eq], &[f.u], &f.pool).unwrap();
    pdsolve(&prob, &PdsolveOptions::default(), &f.pool)
}

#[test]
fn transport_constant_coefficients() {
    let f = fx();
    let p = &f.pool;
    // 2 u_x + 3 u_y = 0
    let eq = p.add(vec![
        p.mul(vec![p.integer(2), dd(&f, f.x)]),
        p.mul(vec![p.integer(3), dd(&f, f.y)]),
    ]);
    let s = solve(&f, eq).unwrap();
    assert_eq!(s.kind, PdeSolutionKind::General);
    assert!(s.verification.status.is_verified());
    assert_eq!(s.arbitrary_functions, vec!["F1".to_string()]);
}

#[test]
fn linear_inhomogeneous_sympy() {
    let f = fx();
    let p = &f.pool;
    // u + u_x + u_y + x*y = 0
    let eq = p.add(vec![f.u, dd(&f, f.x), dd(&f, f.y), p.mul(vec![f.x, f.y])]);
    let s = solve(&f, eq).unwrap();
    assert!(s.verification.status.is_verified());
    assert_eq!(s.method, "first_order_linear");
}

#[test]
fn burgers_cauchy_linear_data() {
    let f = fx();
    let p = &f.pool;
    // u_y + u u_x = 0, u(x, 0) = x  ⇒  u = x/(1 + y)
    let eq = p.add(vec![dd(&f, f.y), p.mul(vec![f.u, dd(&f, f.x)])]);
    let ic = p.func("u", vec![f.x, p.integer(0)]);
    let prob = PdeProblem::new(&[eq], &[f.u], p)
        .unwrap()
        .ic(ic, f.x, p)
        .unwrap();
    let s = pdsolve(&prob, &PdsolveOptions::default(), p).unwrap();
    assert!(s.verification.status.is_verified());
    assert!(!s.implicit, "{}", p.display(s.value));
}

#[test]
fn burgers_breaking_time() {
    let f = fx();
    let p = &f.pool;
    // u_y + u u_x = 0, u(x, 0) = -x  ⇒  breaking at y = 1
    let eq = p.add(vec![dd(&f, f.y), p.mul(vec![f.u, dd(&f, f.x)])]);
    let ic = p.func("u", vec![f.x, p.integer(0)]);
    let prob = PdeProblem::new(&[eq], &[f.u], p)
        .unwrap()
        .ic(ic, p.mul(vec![p.integer(-1), f.x]), p)
        .unwrap();
    let s = pdsolve(&prob, &PdsolveOptions::default(), p).unwrap();
    assert!(s.verification.status.is_verified());
    let shown: Vec<String> = s
        .side_conditions
        .iter()
        .map(|c| c.display_with(p).to_string())
        .collect();
    assert!(
        shown.iter().any(|c| c.contains("> 0") && c.contains('y')),
        "{shown:?}"
    );
}

#[test]
fn characteristic_data_is_refused() {
    let f = fx();
    let p = &f.pool;
    // u_x = 0 with data on y = 0 along x: characteristic.
    let eq = dd(&f, f.x);
    let ic = p.func("u", vec![f.x, p.integer(0)]);
    let prob = PdeProblem::new(&[eq], &[f.u], p)
        .unwrap()
        .ic(ic, f.x, p)
        .unwrap();
    let e = pdsolve(&prob, &PdsolveOptions::default(), p).unwrap_err();
    assert_eq!(e.code(), "E-PDE-052");
}

#[test]
fn eikonal_like_f_p() {
    let f = fx();
    let p = &f.pool;
    // p q = 1
    let eq = p.add(vec![p.mul(vec![dd(&f, f.x), dd(&f, f.y)]), p.integer(-1)]);
    let s = solve(&f, eq).unwrap();
    assert_eq!(s.kind, PdeSolutionKind::CompleteIntegral);
    assert_eq!(s.constants.len(), 2);
}

#[test]
fn clairaut_with_singular_integral() {
    let f = fx();
    let p = &f.pool;
    let (px, py) = (dd(&f, f.x), dd(&f, f.y));
    // u = x p + y q + p q
    let eq = p.add(vec![
        f.u,
        p.mul(vec![p.integer(-1), f.x, px]),
        p.mul(vec![p.integer(-1), f.y, py]),
        p.mul(vec![p.integer(-1), px, py]),
    ]);
    let s = solve(&f, eq).unwrap();
    assert_eq!(s.method, "clairaut");
    assert_eq!(s.alternatives.len(), 1);
    assert_eq!(s.alternatives[0].kind, PdeSolutionKind::SingularIntegral);
}

#[test]
fn hamilton_jacobi_separable() {
    let f = fx();
    let p = &f.pool;
    // u_y + u_x^2 + x^2 = 0  (harmonic-oscillator-like HJ, separable)
    let eq = p.add(vec![
        dd(&f, f.y),
        p.pow(dd(&f, f.x), p.integer(2)),
        p.pow(f.x, p.integer(2)),
    ]);
    let s = solve(&f, eq);
    if let Ok(s) = s {
        assert!(s.verification.status.is_verified());
    }
}

#[test]
fn unknown_hint_and_order_two_refuse() {
    let f = fx();
    let p = &f.pool;
    let prob = PdeProblem::new(&[dd(&f, f.x)], &[f.u], p).unwrap();
    let e = pdsolve(&prob, &PdsolveOptions::default().with_hint("nope"), p).unwrap_err();
    assert_eq!(e.code(), "E-PDE-050");
    let uxx = crate::diff::diff_impl::diff(dd(&f, f.x), f.x, p).unwrap().value;
    let prob2 = PdeProblem::new(&[uxx], &[f.u], p).unwrap();
    let e2 = pdsolve(&prob2, &PdsolveOptions::default(), p).unwrap_err();
    assert_eq!(e2.code(), "E-PDE-050");
}

fn show(f: &Fx, prob: &PdeProblem) -> String {
    match pdsolve(prob, &PdsolveOptions::default(), &f.pool) {
        Ok(s) => format!("OK {} {}", s.method, f.pool.display(s.value)),
        Err(e) => format!("ERR {e}"),
    }
}

#[test]
#[ignore]
fn debug_cases() {
    let f = fx();
    let p = &f.pool;
    let (px, py) = (dd(&f, f.x), dd(&f, f.y));
    let m = |a: Vec<ExprId>| p.mul(a);
    let i = |n: i64| p.integer(n);
    // -2 u_x + 4 u_y + 5 u - exp(x + 3y)
    let e1 = p.add(vec![
        m(vec![i(-2), px]),
        m(vec![i(4), py]),
        m(vec![i(5), f.u]),
        m(vec![i(-1), p.func("exp", vec![p.add(vec![f.x, m(vec![i(3), f.y])])])]),
    ]);
    println!("ccoeff-1: {}", show(&f, &PdeProblem::new(&[e1], &[f.u], p).unwrap()));
    // p^2 + q^2 - x - y
    let e2 = p.add(vec![
        p.pow(px, i(2)),
        p.pow(py, i(2)),
        m(vec![i(-1), f.x]),
        m(vec![i(-1), f.y]),
    ]);
    println!("sep-2: {}", show(&f, &PdeProblem::new(&[e2], &[f.u], p).unwrap()));
    let pr2 = PdeProblem::new(&[e2], &[f.u], p).unwrap();
    match pdsolve(&pr2, &PdsolveOptions::default().with_hint("separable"), p) {
        Ok(s) => println!("sep-2 hint: OK {}", p.display(s.value)),
        Err(e) => println!("sep-2 hint: ERR {e}"),
    }
    // u_y + 2 u_x, u(x,0) = exp(-x^2)
    let e3 = p.add(vec![py, m(vec![i(2), px])]);
    let ic = p.func("u", vec![f.x, i(0)]);
    let g = p.func("exp", vec![m(vec![i(-1), p.pow(f.x, i(2))])]);
    let pr = PdeProblem::new(&[e3], &[f.u], p).unwrap().ic(ic, g, p).unwrap();
    println!("advection: {}", show(&f, &pr));
    // u_y + x u_x, u(x,0) = sin x
    let e4 = p.add(vec![py, m(vec![f.x, px])]);
    let pr = PdeProblem::new(&[e4], &[f.u], p)
        .unwrap()
        .ic(ic, p.func("sin", vec![f.x]), p)
        .unwrap();
    println!("stretch: {}", show(&f, &pr));
    // Burgers sine
    let e5 = p.add(vec![py, m(vec![f.u, px])]);
    let pr = PdeProblem::new(&[e5], &[f.u], p)
        .unwrap()
        .ic(ic, p.func("sin", vec![f.x]), p)
        .unwrap();
    println!("burgers-sine: {}", show(&f, &pr));
}
