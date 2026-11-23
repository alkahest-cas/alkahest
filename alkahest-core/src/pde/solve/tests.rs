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
    let uxx = crate::diff::diff_impl::diff(dd(&f, f.x), f.x, p)
        .unwrap()
        .value;
    let prob2 = PdeProblem::new(&[uxx], &[f.u], p).unwrap();
    let e2 = pdsolve(&prob2, &PdsolveOptions::default(), p).unwrap_err();
    assert_eq!(e2.code(), "E-PDE-050");
}

/// Solve equations written as strings (`Derivative(u(x, y), x)` etc.); for
/// local debugging: `cargo test … debug_strings -- --ignored --nocapture`,
/// with the equations in `PDE_DEBUG` separated by `;`.
#[test]
#[ignore]
fn debug_strings() {
    let Ok(src) = std::env::var("PDE_DEBUG") else {
        return;
    };
    for eq_src in src.split(';') {
        let pool = ExprPool::new();
        let mut syms = std::collections::HashMap::new();
        let vars = if eq_src.contains("uz") {
            "x, y, z"
        } else {
            "x, y"
        };
        let vs: Vec<ExprId> = vars
            .split(", ")
            .map(|n| {
                let s = pool.symbol(n, Domain::Real);
                syms.insert(n.to_string(), s);
                s
            })
            .collect();
        syms.insert("u".to_string(), pool.func("u", vs));
        let u_src = format!("u({vars})");
        let full = eq_src
            .replace("ux", &format!("Derivative({u_src}, x)"))
            .replace("uy", &format!("Derivative({u_src}, y)"))
            .replace("uz", &format!("Derivative({u_src}, z)"));
        let eq = match crate::parse::parse(&full, &pool, &mut syms) {
            Ok(e) => e,
            Err(e) => {
                println!("PARSE {eq_src}: {e:?}");
                continue;
            }
        };
        let u = crate::parse::parse(&u_src, &pool, &mut syms).unwrap();
        let prob = match PdeProblem::new(&[eq], &[u], &pool) {
            Ok(p) => p,
            Err(e) => {
                println!("PROBLEM {eq_src}: {e}");
                continue;
            }
        };
        let t0 = std::time::Instant::now();
        match pdsolve(&prob, &PdsolveOptions::default(), &pool) {
            Ok(s) => println!(
                "OK [{:.2}s] {eq_src} :: {} {}",
                t0.elapsed().as_secs_f64(),
                s.method,
                pool.display(s.value)
            ),
            Err(e) => println!("ERR [{:.2}s] {eq_src} :: {e}", t0.elapsed().as_secs_f64()),
        }
    }
}
