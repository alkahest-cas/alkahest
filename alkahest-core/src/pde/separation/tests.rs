use std::collections::HashMap;

use super::*;
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::pde::{PdeDomain, PdeProblem, VerificationStatus};

struct Pb {
    pool: ExprPool,
    syms: HashMap<String, ExprId>,
}

impl Pb {
    /// Symbols `x, y, z, t, theta, phi` real; `L, a, b, c, k, R, h` positive;
    /// `r` positive; unknown `u(args…)`.
    fn new(args: &[&str]) -> Self {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        for n in ["x", "y", "z", "t", "theta", "phi", "s"] {
            syms.insert(n.to_string(), pool.symbol(n, Domain::Real));
        }
        for n in ["L", "a", "b", "c", "k", "R", "h", "r", "T0", "T1", "A", "B"] {
            syms.insert(n.to_string(), pool.symbol(n, Domain::Positive));
        }
        let a: Vec<ExprId> = args.iter().map(|n| syms[*n]).collect();
        let u = pool.func("u", a);
        syms.insert("u".to_string(), u);
        Pb { pool, syms }
    }
    fn e(&mut self, s: &str) -> ExprId {
        crate::parse::parse(s, &self.pool, &mut self.syms).unwrap_or_else(|e| panic!("{s}: {e}"))
    }
    fn v(&self, n: &str) -> ExprId {
        self.syms[n]
    }
    fn interval(&mut self, v: &str, lo: &str, hi: &str) -> PdeDomain {
        let (lo, hi) = (self.e(lo), self.e(hi));
        PdeDomain::interval(self.v(v), lo, hi, &self.pool).unwrap()
    }
    fn time(&self, v: &str) -> PdeDomain {
        PdeDomain::half_line(self.v(v), self.pool.integer(0), false, &self.pool).unwrap()
    }
    fn product(&self, ps: Vec<PdeDomain>) -> PdeDomain {
        PdeDomain::product(ps, &self.pool).unwrap()
    }
    fn problem(
        &mut self,
        eq: &str,
        dom: PdeDomain,
        bcs: &[(&str, &str)],
        ics: &[(&str, &str)],
    ) -> PdeProblem {
        let eqe = self.e(eq);
        let u = self.syms["u"];
        let mut p = PdeProblem::new(&[eqe], &[u], &self.pool)
            .unwrap()
            .on(dom, &self.pool)
            .unwrap();
        for (l, r) in bcs {
            let (l, r) = (self.e(l), self.e(r));
            p = p.bc(l, r, &self.pool).unwrap();
        }
        for (l, r) in ics {
            let (l, r) = (self.e(l), self.e(r));
            p = p.ic(l, r, &self.pool).unwrap();
        }
        p
    }
    fn show(&self, e: ExprId) -> String {
        self.pool.display(e).to_string()
    }
}

fn verified(sol: &SeriesSolution, pb: &Pb) {
    for c in &sol.checks {
        assert_eq!(
            c.status,
            crate::simplify::zero::ZeroStatus::ProvedZero,
            "check {} ({}) of group {}: {:?} — {}\nvalue = {}",
            c.what,
            c.method,
            c.group,
            c.status,
            c.detail,
            pb.show(sol.value)
        );
    }
    assert_eq!(sol.status, VerificationStatus::ExactlyVerified);
    for pc in &sol.projection {
        assert_ne!(pc.status, VerificationStatus::Refuted, "{:?}", pc.evidence);
    }
}

#[test]
fn separates_heat_and_polar_laplacian() {
    let mut pb = Pb::new(&["x", "t"]);
    let eq = pb.e("Derivative(u, t) - k*Derivative(u, x, x)");
    let u = pb.v("u");
    let s = separate(eq, u, SeparationKind::Product, &pb.pool).unwrap();
    assert_eq!(s.equations.len(), 2);
    assert_eq!(s.constants.len(), 1);
    let shown: Vec<String> = s.equations.iter().map(|e| pb.show(e.equation)).collect();
    assert!(shown.iter().any(|x| x.contains("mu1")), "{shown:?}");
    // Polar heat: θ needs the multiplier r².
    let mut pb = Pb::new(&["r", "theta", "t"]);
    let eq = pb.e(
        "Derivative(u, t) - k*(Derivative(u, r, r) + Derivative(u, r)/r + Derivative(u, theta, \
         theta)/r^2)",
    );
    let u = pb.v("u");
    let s = separate(eq, u, SeparationKind::Product, &pb.pool).unwrap();
    assert_eq!(s.equations.len(), 3);
    assert_eq!(s.equations[0].var, pb.v("theta"));
    // Not separable: u_t = u_xx + x t u.
    let mut pb = Pb::new(&["x", "t"]);
    let eq = pb.e("Derivative(u, t) - Derivative(u, x, x) - (x + t)^2*u");
    let u = pb.v("u");
    let err = separate(eq, u, SeparationKind::Product, &pb.pool).unwrap_err();
    assert_eq!(err.code(), "E-PDE-200");
    // Additive: u_x^2 + u_t = 0.
    let mut pb = Pb::new(&["x", "t"]);
    let eq = pb.e("Derivative(u, x)^2 + Derivative(u, t)");
    let u = pb.v("u");
    let s = separate(eq, u, SeparationKind::Additive, &pb.pool).unwrap();
    assert_eq!(s.equations.len(), 2);
}

#[test]
fn heat_dirichlet_interval() {
    let mut pb = Pb::new(&["x", "t"]);
    let dom = {
        let i = pb.interval("x", "0", "L");
        let t = pb.time("t");
        pb.product(vec![i, t])
    };
    let prob = pb.problem(
        "Derivative(u, t) - k*Derivative(u, x, x)",
        dom,
        &[("Subs(u, x, 0)", "0"), ("Subs(u, x, L)", "0")],
        &[("Subs(u, t, 0)", "x*(L - x)")],
    );
    let sol = separation_solve(&prob, &pb.pool).unwrap();
    eprintln!("heat: {}", pb.show(sol.value));
    verified(&sol, &pb);
    assert_eq!(sol.kind, "series");
    assert_eq!(sol.convergence, "uniform");
    assert!(!sol.projection.is_empty());
}

#[test]
fn wave_and_finite_data() {
    let mut pb = Pb::new(&["x", "t"]);
    let dom = {
        let i = pb.interval("x", "0", "pi");
        let t = pb.time("t");
        pb.product(vec![i, t])
    };
    let prob = pb.problem(
        "Derivative(u, t, t) - c^2*Derivative(u, x, x)",
        dom,
        &[("Subs(u, x, 0)", "0"), ("Subs(u, x, pi)", "0")],
        &[
            ("Subs(u, t, 0)", "sin(x) + 3*sin(2*x)"),
            ("Subs(Derivative(u, t), t, 0)", "0"),
        ],
    );
    let sol = separation_solve(&prob, &pb.pool).unwrap();
    eprintln!("wave: {}", pb.show(sol.value));
    verified(&sol, &pb);
    assert_eq!(sol.kind, "closed_form", "{}", pb.show(sol.value));
}

#[test]
fn laplace_rectangle_and_disk() {
    let mut pb = Pb::new(&["x", "y"]);
    let (i, j) = (pb.interval("x", "0", "a"), pb.interval("y", "0", "b"));
    let dom = pb.product(vec![i, j]);
    let prob = pb.problem(
        "Derivative(u, x, x) + Derivative(u, y, y)",
        dom,
        &[
            ("Subs(u, x, 0)", "0"),
            ("Subs(u, x, a)", "0"),
            ("Subs(u, y, 0)", "0"),
            ("Subs(u, y, b)", "x"),
        ],
        &[],
    );
    let sol = separation_solve(&prob, &pb.pool).unwrap();
    eprintln!("laplace rect: {}", pb.show(sol.value));
    verified(&sol, &pb);
    assert_eq!(sol.convergence, "l2");

    let mut pb = Pb::new(&["r", "theta"]);
    let (r, th, rr) = (pb.v("r"), pb.v("theta"), pb.v("R"));
    let dom = PdeDomain::polar_disk(r, Some(th), rr, &pb.pool).unwrap();
    let prob = pb.problem(
        "Derivative(u, r, r) + Derivative(u, r)/r + Derivative(u, theta, theta)/r^2",
        dom,
        &[("Subs(u, r, R)", "cos(theta)^2")],
        &[],
    );
    let sol = separation_solve(&prob, &pb.pool).unwrap();
    eprintln!("laplace disk: {}", pb.show(sol.value));
    verified(&sol, &pb);
    assert_eq!(sol.kind, "closed_form");
    assert!(sol
        .alternatives
        .iter()
        .any(|(n, _)| n == "poisson_integral"));
}

#[test]
fn heat_with_constant_boundary_data_uses_steady_state() {
    let mut pb = Pb::new(&["x", "t"]);
    let dom = {
        let i = pb.interval("x", "0", "L");
        let t = pb.time("t");
        pb.product(vec![i, t])
    };
    let prob = pb.problem(
        "Derivative(u, t) - k*Derivative(u, x, x)",
        dom,
        &[("Subs(u, x, 0)", "T0"), ("Subs(u, x, L)", "T1")],
        &[("Subs(u, t, 0)", "0")],
    );
    let sol = separation_solve(&prob, &pb.pool).unwrap();
    eprintln!("steady: {} | {}", pb.show(sol.value), sol.method);
    verified(&sol, &pb);
    assert!(sol.steady_state.is_some());
    assert!(sol.method.contains("steady_state"));
}

#[test]
fn heat_disk_radial_bessel() {
    let mut pb = Pb::new(&["r", "t"]);
    let (r, one) = (pb.v("r"), pb.pool.integer(1));
    let disk = PdeDomain::polar_disk(r, None, one, &pb.pool).unwrap();
    let t = pb.time("t");
    let dom = pb.product(vec![disk, t]);
    let prob = pb.problem(
        "Derivative(u, t) - (Derivative(u, r, r) + Derivative(u, r)/r)",
        dom,
        &[("Subs(u, r, 1)", "0")],
        &[("Subs(u, t, 0)", "1 - r^2")],
    );
    let sol = separation_solve(&prob, &pb.pool).unwrap();
    eprintln!("bessel: {}", pb.show(sol.value));
    verified(&sol, &pb);
}

#[test]
fn poisson_square_double_series() {
    let mut pb = Pb::new(&["x", "y"]);
    let (i, j) = (pb.interval("x", "0", "pi"), pb.interval("y", "0", "pi"));
    let dom = pb.product(vec![i, j]);
    let prob = pb.problem(
        "Derivative(u, x, x) + Derivative(u, y, y) + 1",
        dom,
        &[
            ("Subs(u, x, 0)", "0"),
            ("Subs(u, x, pi)", "0"),
            ("Subs(u, y, 0)", "0"),
            ("Subs(u, y, pi)", "0"),
        ],
        &[],
    );
    let sol = separation_solve(&prob, &pb.pool).unwrap();
    eprintln!("poisson: {}", pb.show(sol.value));
    verified(&sol, &pb);
}

#[test]
fn error_codes_are_registered() {
    use crate::pde::PdeError as E;
    for e in [
        E::SeparationNotSeparable(String::new()),
        E::SeparationUnsupported(String::new()),
        E::SeparationCondition(String::new()),
        E::SeparationMissingData(String::new()),
        E::SeparationModeOde(String::new()),
        E::SeparationLifting(String::new()),
        E::SeparationVerificationFailed(String::new()),
        E::SeparationCompatibility(String::new()),
        E::SeparationProjectionRefuted(String::new()),
        E::SeparationTruncation(String::new()),
    ] {
        let n: u32 = e.code().strip_prefix("E-PDE-").unwrap().parse().unwrap();
        assert!((200..=249).contains(&n));
        assert!(e.remediation().is_some(), "{} not registered", e.code());
    }
}

fn heat_on(pb: &mut Pb, lo: &str, hi: &str) -> PdeDomain {
    let i = pb.interval("x", lo, hi);
    let t = pb.time("t");
    pb.product(vec![i, t])
}

fn solve_show(pb: &Pb, prob: &PdeProblem, tag: &str) -> SeriesSolution {
    let sol = separation_solve(prob, &pb.pool).unwrap_or_else(|e| panic!("{tag}: {e}"));
    eprintln!(
        "{tag}: {} | {} | {:?}",
        pb.show(sol.value),
        sol.method,
        sol.notes
    );
    sol
}

#[test]
fn neumann_robin_periodic() {
    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "0", "L");
    let prob = pb.problem(
        "Derivative(u, t) - k*Derivative(u, x, x)",
        dom,
        &[
            ("Subs(Derivative(u, x), x, 0)", "0"),
            ("Subs(Derivative(u, x), x, L)", "0"),
        ],
        &[("Subs(u, t, 0)", "x")],
    );
    let sol = solve_show(&pb, &prob, "neumann");
    verified(&sol, &pb);

    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "0", "1");
    let prob = pb.problem(
        "Derivative(u, t) - Derivative(u, x, x)",
        dom,
        &[
            ("Subs(u, x, 0)", "0"),
            ("Subs(Derivative(u, x), x, 1) + Subs(u, x, 1)", "0"),
        ],
        &[("Subs(u, t, 0)", "1")],
    );
    let sol = solve_show(&pb, &prob, "robin");
    verified(&sol, &pb);
    assert!(!sol.projection.is_empty());

    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "-pi", "pi");
    let prob = pb.problem(
        "Derivative(u, t) - Derivative(u, x, x)",
        dom,
        &[
            ("Subs(u, x, -pi)", "Subs(u, x, pi)"),
            (
                "Subs(Derivative(u, x), x, -pi)",
                "Subs(Derivative(u, x), x, pi)",
            ),
        ],
        &[("Subs(u, t, 0)", "x^2")],
    );
    let sol = solve_show(&pb, &prob, "periodic");
    verified(&sol, &pb);
}

#[test]
fn duhamel_and_source() {
    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "0", "1");
    let prob = pb.problem(
        "Derivative(u, t) - Derivative(u, x, x)",
        dom,
        &[("Subs(u, x, 0)", "0"), ("Subs(u, x, 1)", "sin(t)")],
        &[("Subs(u, t, 0)", "0")],
    );
    let sol = solve_show(&pb, &prob, "duhamel");
    verified(&sol, &pb);
    assert!(sol.method.contains("lifting"));

    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "0", "1");
    let prob = pb.problem(
        "Derivative(u, t) - Derivative(u, x, x) - sin(pi*x)",
        dom,
        &[("Subs(u, x, 0)", "0"), ("Subs(u, x, 1)", "0")],
        &[("Subs(u, t, 0)", "0")],
    );
    let sol = solve_show(&pb, &prob, "source");
    verified(&sol, &pb);
    assert_eq!(sol.kind, "closed_form");
}

#[test]
fn telegraph_schrodinger_helmholtz() {
    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "0", "pi");
    let prob = pb.problem(
        "Derivative(u, t, t) + 2*Derivative(u, t) - Derivative(u, x, x)",
        dom,
        &[("Subs(u, x, 0)", "0"), ("Subs(u, x, pi)", "0")],
        &[
            ("Subs(u, t, 0)", "sin(2*x)"),
            ("Subs(Derivative(u, t), t, 0)", "0"),
        ],
    );
    let sol = solve_show(&pb, &prob, "telegraph");
    verified(&sol, &pb);

    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "0", "pi");
    let prob = pb.problem(
        "I*Derivative(u, t) + Derivative(u, x, x)",
        dom,
        &[("Subs(u, x, 0)", "0"), ("Subs(u, x, pi)", "0")],
        &[("Subs(u, t, 0)", "sin(x)")],
    );
    let sol = solve_show(&pb, &prob, "schrodinger");
    verified(&sol, &pb);

    let mut pb = Pb::new(&["x", "y"]);
    let (i, j) = (pb.interval("x", "0", "pi"), pb.interval("y", "0", "1"));
    let dom = pb.product(vec![i, j]);
    let prob = pb.problem(
        "Derivative(u, x, x) + Derivative(u, y, y) + 2*u",
        dom,
        &[
            ("Subs(u, x, 0)", "0"),
            ("Subs(u, x, pi)", "0"),
            ("Subs(u, y, 0)", "0"),
            ("Subs(u, y, 1)", "sin(3*x)"),
        ],
        &[],
    );
    let sol = solve_show(&pb, &prob, "helmholtz");
    verified(&sol, &pb);
}

#[test]
fn curvilinear_domains() {
    use crate::pde::CoordinateSystem as C;
    // Laplace on an annulus.
    let mut pb = Pb::new(&["r", "theta"]);
    let (r, th) = (pb.v("r"), pb.v("theta"));
    let (one, two, pi) = (pb.pool.integer(1), pb.pool.integer(2), pb.e("pi"));
    let mpi = pb.e("-pi");
    let dom = PdeDomain::curvilinear(
        C::Polar,
        &[Some(r), Some(th)],
        &[(one, two), (mpi, pi)],
        &pb.pool,
    )
    .unwrap();
    let prob = pb.problem(
        "Derivative(u, r, r) + Derivative(u, r)/r + Derivative(u, theta, theta)/r^2",
        dom,
        &[("Subs(u, r, 1)", "0"), ("Subs(u, r, 2)", "sin(theta)")],
        &[],
    );
    let sol = solve_show(&pb, &prob, "annulus");
    verified(&sol, &pb);
    assert_eq!(sol.kind, "closed_form");

    // Sector 0 < θ < π/2.
    let mut pb = Pb::new(&["r", "theta"]);
    let (r, th) = (pb.v("r"), pb.v("theta"));
    let (zero, one, hp) = (pb.pool.integer(0), pb.pool.integer(1), pb.e("pi/2"));
    let dom = PdeDomain::curvilinear(
        C::Polar,
        &[Some(r), Some(th)],
        &[(zero, one), (zero, hp)],
        &pb.pool,
    )
    .unwrap();
    let prob = pb.problem(
        "Derivative(u, r, r) + Derivative(u, r)/r + Derivative(u, theta, theta)/r^2",
        dom,
        &[
            ("Subs(u, theta, 0)", "0"),
            ("Subs(u, theta, pi/2)", "0"),
            ("Subs(u, r, 1)", "theta*(pi/2 - theta)"),
        ],
        &[],
    );
    let sol = solve_show(&pb, &prob, "sector");
    verified(&sol, &pb);

    // Ball, axisymmetric Laplace: finite Legendre expansion.
    let mut pb = Pb::new(&["r", "theta"]);
    let (r, th) = (pb.v("r"), pb.v("theta"));
    let (zero, one, pi, mpi) = (
        pb.pool.integer(0),
        pb.pool.integer(1),
        pb.e("pi"),
        pb.e("-pi"),
    );
    let dom = PdeDomain::curvilinear(
        C::Spherical,
        &[Some(r), Some(th), None],
        &[(zero, one), (zero, pi), (mpi, pi)],
        &pb.pool,
    )
    .unwrap();
    let prob = pb.problem(
        "Derivative(u, r, r) + 2*Derivative(u, r)/r + Derivative(u, theta, theta)/r^2 + cos(theta)/sin(theta)*Derivative(u, theta)/r^2",
        dom,
        &[("Subs(u, r, 1)", "cos(theta)^2")],
        &[],
    );
    let sol = solve_show(&pb, &prob, "ball laplace");
    verified(&sol, &pb);
    assert_eq!(sol.kind, "closed_form");

    // Ball, radial heat.
    let mut pb = Pb::new(&["r", "t"]);
    let r = pb.v("r");
    let (zero, one, pi, mpi) = (
        pb.pool.integer(0),
        pb.pool.integer(1),
        pb.e("pi"),
        pb.e("-pi"),
    );
    let ball = PdeDomain::curvilinear(
        C::Spherical,
        &[Some(r), None, None],
        &[(zero, one), (zero, pi), (mpi, pi)],
        &pb.pool,
    )
    .unwrap();
    let t = pb.time("t");
    let dom = pb.product(vec![ball, t]);
    let prob = pb.problem(
        "Derivative(u, t) - Derivative(u, r, r) - 2*Derivative(u, r)/r",
        dom,
        &[("Subs(u, r, 1)", "0")],
        &[("Subs(u, t, 0)", "1")],
    );
    let sol = solve_show(&pb, &prob, "ball heat");
    verified(&sol, &pb);

    // Cylinder, axisymmetric heat.
    let mut pb = Pb::new(&["r", "z", "t"]);
    let (r, z) = (pb.v("r"), pb.v("z"));
    let (zero, one, pi, mpi) = (
        pb.pool.integer(0),
        pb.pool.integer(1),
        pb.e("pi"),
        pb.e("-pi"),
    );
    let cyl = PdeDomain::curvilinear(
        C::Cylindrical,
        &[Some(r), None, Some(z)],
        &[(zero, one), (mpi, pi), (zero, one)],
        &pb.pool,
    )
    .unwrap();
    let t = pb.time("t");
    let dom = pb.product(vec![cyl, t]);
    let prob = pb.problem(
        "Derivative(u, t) - Derivative(u, r, r) - Derivative(u, r)/r - Derivative(u, z, z)",
        dom,
        &[
            ("Subs(u, r, 1)", "0"),
            ("Subs(u, z, 0)", "0"),
            ("Subs(u, z, 1)", "0"),
        ],
        &[("Subs(u, t, 0)", "(1 - r^2)*sin(pi*z)")],
    );
    let sol = solve_show(&pb, &prob, "cylinder");
    verified(&sol, &pb);

    // Cartesian disk.
    let mut pb = Pb::new(&["x", "y", "t"]);
    let (x, y) = (pb.v("x"), pb.v("y"));
    let z0 = pb.pool.integer(0);
    let one = pb.pool.integer(1);
    let disk = PdeDomain::disk([x, y], [z0, z0], one, &pb.pool).unwrap();
    let t = pb.time("t");
    let dom = pb.product(vec![disk, t]);
    let prob = pb.problem(
        "Derivative(u, t) - Derivative(u, x, x) - Derivative(u, y, y)",
        dom,
        &[("u", "0")],
        &[("Subs(u, t, 0)", "1 - x^2 - y^2")],
    );
    let sol = solve_show(&pb, &prob, "cartesian disk");
    verified(&sol, &pb);
    assert!(sol.coordinates.is_some());
}

#[test]
fn symbolic_and_piecewise_data() {
    let mut pb = Pb::new(&["x", "t"]);
    let f = pb.pool.func("f", vec![pb.v("x")]);
    pb.syms.insert("f".into(), f);
    let g = pb.pool.func("g", vec![pb.v("t")]);
    pb.syms.insert("g".into(), g);
    let dom = heat_on(&mut pb, "0", "L");
    let prob = pb.problem(
        "Derivative(u, t) - k*Derivative(u, x, x)",
        dom,
        &[("Subs(u, x, 0)", "0"), ("Subs(u, x, L)", "0")],
        &[("Subs(u, t, 0)", "f")],
    );
    let sol = solve_show(&pb, &prob, "symbolic f");
    verified(&sol, &pb);
    assert!(crate::pde::separation::has_integral(sol.value, &pb.pool));

    // Piecewise data (Haberman 2.4.1(a)).
    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "0", "L");
    let x = pb.v("x");
    let l2 = pb.e("L/2");
    let cond = pb
        .pool
        .predicate(crate::kernel::expr::PredicateKind::Lt, vec![x, l2]);
    let pw = pb
        .pool
        .piecewise(vec![(cond, pb.pool.integer(0))], pb.pool.integer(1));
    pb.syms.insert("pw".into(), pw);
    let prob = pb.problem(
        "Derivative(u, t) - k*Derivative(u, x, x)",
        dom,
        &[
            ("Subs(Derivative(u, x), x, 0)", "0"),
            ("Subs(Derivative(u, x), x, L)", "0"),
        ],
        &[("Subs(u, t, 0)", "pw")],
    );
    let sol = solve_show(&pb, &prob, "piecewise");
    verified(&sol, &pb);
    assert!(!crate::pde::separation::has_integral(sol.value, &pb.pool));

    // Time-dependent symbolic boundary data: Duhamel kept as an integral,
    // verified through Leibniz's rule.
    let mut pb = Pb::new(&["x", "t"]);
    let g = pb.pool.func("g", vec![pb.v("t")]);
    pb.syms.insert("g".into(), g);
    let dom = heat_on(&mut pb, "0", "1");
    let prob = pb.problem(
        "Derivative(u, t) - Derivative(u, x, x)",
        dom,
        &[("Subs(u, x, 0)", "0"), ("Subs(u, x, 1)", "g")],
        &[("Subs(u, t, 0)", "0")],
    );
    let sol = solve_show(&pb, &prob, "duhamel g");
    verified(&sol, &pb);
}

#[test]
fn drum_with_piecewise_velocity() {
    use crate::pde::CoordinateSystem as C;
    let mut pb = Pb::new(&["r", "theta", "t"]);
    let (r, th) = (pb.v("r"), pb.v("theta"));
    let (zero, one, pi, mpi) = (
        pb.pool.integer(0),
        pb.pool.integer(1),
        pb.e("pi"),
        pb.e("-pi"),
    );
    let disk = PdeDomain::curvilinear(
        C::Polar,
        &[Some(r), Some(th)],
        &[(zero, one), (mpi, pi)],
        &pb.pool,
    )
    .unwrap();
    let t = pb.time("t");
    let dom = pb.product(vec![disk, t]);
    let half = pb.e("1/2");
    let cond = pb
        .pool
        .predicate(crate::kernel::expr::PredicateKind::Lt, vec![r, half]);
    let val = pb.e("4/pi");
    let pw = pb.pool.piecewise(vec![(cond, val)], pb.pool.integer(0));
    pb.syms.insert("pw".into(), pw);
    let prob = pb.problem(
        "Derivative(u, t, t) - (Derivative(u, r, r) + Derivative(u, r)/r + Derivative(u, theta, theta)/r^2)",
        dom,
        &[("Subs(u, r, 1)", "0")],
        &[("Subs(u, t, 0)", "0"), ("Subs(Derivative(u, t), t, 0)", "pw")],
    );
    let sol = solve_show(&pb, &prob, "drum piecewise");
    verified(&sol, &pb);
}

#[test]
fn exceptional_indices_with_parameters_are_expanded() {
    // Haberman 2.3.3(b): finitely many modes, L symbolic — the generic
    // coefficient is 0/0 at n = 1, 3, which must not be read as 0.
    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "0", "L");
    let prob = pb.problem(
        "Derivative(u, t) - k*Derivative(u, x, x)",
        dom,
        &[("Subs(u, x, 0)", "0"), ("Subs(u, x, L)", "0")],
        &[("Subs(u, t, 0)", "3*sin(pi*x/L) - sin(3*pi*x/L)")],
    );
    let sol = solve_show(&pb, &prob, "haberman 2.3.3b");
    verified(&sol, &pb);
    assert_eq!(sol.kind, "closed_form");
    assert!(pb.show(sol.value).contains("sin"), "{}", pb.show(sol.value));
    // Haberman 2.3.3(c): 2 cos(3πx/L) — the n = 3 term on its own.
    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "0", "L");
    let prob = pb.problem(
        "Derivative(u, t) - k*Derivative(u, x, x)",
        dom,
        &[("Subs(u, x, 0)", "0"), ("Subs(u, x, L)", "0")],
        &[("Subs(u, t, 0)", "2*cos(3*pi*x/L)")],
    );
    let sol = solve_show(&pb, &prob, "haberman 2.3.3c");
    verified(&sol, &pb);
    assert!(sol.modes.len() >= 2);
    // Pinchover–Rubinstein 6.21: a source t cos(2001x) — the range of n is
    // split around n = 2001 instead of expanding 2001 terms.
    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "0", "pi");
    let prob = pb.problem(
        "Derivative(u, t) - Derivative(u, x, x) - t*cos(2001*x)",
        dom,
        &[
            ("Subs(Derivative(u, x), x, 0)", "0"),
            ("Subs(Derivative(u, x), x, pi)", "0"),
        ],
        &[("Subs(u, t, 0)", "pi*cos(2*x)")],
    );
    let sol = solve_show(&pb, &prob, "pinchover 6.21");
    verified(&sol, &pb);
    assert_eq!(sol.kind, "closed_form");
}

#[test]
fn convection_term_removed_by_exponential_shift() {
    // u_t = u_xx + 2u_x on (0, π), Dirichlet, u(x, 0) = e^{−x} sin x: the
    // direction X'' + 2X' + μX = 0 becomes Y'' + (μ − 1)Y = 0 with
    // X = e^{−x}Y; the solution is e^{−x} sin x · e^{−2t}.
    let mut pb = Pb::new(&["x", "t"]);
    let dom = heat_on(&mut pb, "0", "pi");
    let prob = pb.problem(
        "Derivative(u, t) - Derivative(u, x, x) - 2*Derivative(u, x)",
        dom,
        &[("Subs(u, x, 0)", "0"), ("Subs(u, x, pi)", "0")],
        &[("Subs(u, t, 0)", "exp(-x)*sin(x)")],
    );
    let sol = solve_show(&pb, &prob, "convection");
    verified(&sol, &pb);
    assert!(
        sol.notes
            .iter()
            .any(|n| n.contains("first-derivative term")),
        "{:?}",
        sol.notes
    );
    if sol.kind == "closed_form" {
        let expect = pb.e("exp(-x)*sin(x)*exp(-2*t)");
        let diff = crate::simplify::engine::simplify(
            pb.pool.add(vec![
                sol.value,
                pb.pool.mul(vec![pb.pool.integer(-1), expect]),
            ]),
            &pb.pool,
        )
        .value;
        assert_eq!(
            crate::simplify::zero::zero_test(diff, &pb.pool, None).status,
            crate::simplify::zero::ZeroStatus::ProvedZero,
            "{}",
            pb.show(sol.value)
        );
    }
}
