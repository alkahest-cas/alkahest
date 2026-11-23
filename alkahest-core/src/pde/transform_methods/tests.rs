use super::*;
use crate::errors::AlkahestError;

struct P {
    pool: ExprPool,
}

impl P {
    fn new() -> Self {
        P {
            pool: ExprPool::new(),
        }
    }
    fn x(&self) -> ExprId {
        self.pool.symbol("x", Domain::Real)
    }
    fn t(&self) -> ExprId {
        self.pool.symbol("t", Domain::Positive)
    }
    fn u(&self) -> ExprId {
        self.pool.func("u", vec![self.x(), self.t()])
    }
    fn dx(&self, e: ExprId, v: ExprId, k: usize) -> ExprId {
        let mut e = e;
        for _ in 0..k {
            e = crate::diff::diff(e, v, &self.pool).unwrap().value;
        }
        e
    }
    fn at(&self, e: ExprId, v: ExprId, a: i64) -> ExprId {
        subs1(e, v, self.pool.integer(a), &self.pool)
    }
    fn line(&self) -> PdeDomain {
        let p = &self.pool;
        PdeDomain::product(
            vec![
                PdeDomain::line(self.x(), p).unwrap(),
                PdeDomain::half_line(self.t(), p.integer(0), false, p).unwrap(),
            ],
            p,
        )
        .unwrap()
    }
    fn quadrant(&self) -> PdeDomain {
        let p = &self.pool;
        PdeDomain::product(
            vec![
                PdeDomain::half_line(self.x(), p.integer(0), false, p).unwrap(),
                PdeDomain::half_line(self.t(), p.integer(0), false, p).unwrap(),
            ],
            p,
        )
        .unwrap()
    }
    /// u_t − k u_xx
    fn heat(&self, k: ExprId) -> ExprId {
        let p = &self.pool;
        let u = self.u();
        p.add(vec![
            self.dx(u, self.t(), 1),
            p.mul(vec![p.integer(-1), k, self.dx(u, self.x(), 2)]),
        ])
    }
    /// u_tt − c² u_xx
    fn wave(&self, c: ExprId) -> ExprId {
        let p = &self.pool;
        let u = self.u();
        p.add(vec![
            self.dx(u, self.t(), 2),
            p.mul(vec![
                p.integer(-1),
                p.pow(c, p.integer(2)),
                self.dx(u, self.x(), 2),
            ]),
        ])
    }
}

fn shown(e: ExprId, pool: &ExprPool) -> String {
    pool.display(e).to_string()
}

fn exact(sol: &TransformSolution, pool: &ExprPool) {
    assert_eq!(
        sol.status,
        VerificationStatus::ExactlyVerified,
        "{} {:?}",
        shown(sol.value, pool),
        sol.checks
    );
}

#[test]
fn heat_on_the_line_symbolic_data_is_a_heat_kernel_integral() {
    let s = P::new();
    let p = &s.pool;
    let k = p.symbol("k", Domain::Positive);
    let f = p.func("f", vec![s.x()]);
    let prob = PdeProblem::new(&[s.heat(k)], &[s.u()], p)
        .unwrap()
        .on(s.line(), p)
        .unwrap()
        .ic(s.at(s.u(), s.t(), 0), f, p)
        .unwrap();
    let sol = fourier_method(&prob, p).unwrap();
    assert_eq!(sol.kind, SolutionKind::IntegralRepresentation);
    exact(&sol, p);
    let v = shown(sol.value, p);
    assert!(
        v.contains("Integral") && v.contains("exp") && v.contains("f(s)"),
        "{v}"
    );
}

#[test]
fn heat_with_gaussian_data_closes() {
    let s = P::new();
    let p = &s.pool;
    let x = s.x();
    let f = p.func(
        "exp",
        vec![p.mul(vec![p.integer(-1), p.pow(x, p.integer(2))])],
    );
    let prob = PdeProblem::new(&[s.heat(p.integer(1))], &[s.u()], p)
        .unwrap()
        .on(s.line(), p)
        .unwrap()
        .ic(s.at(s.u(), s.t(), 0), f, p)
        .unwrap();
    let sol = fourier_method(&prob, p).unwrap();
    assert_eq!(
        sol.kind,
        SolutionKind::ClosedForm,
        "{}",
        shown(sol.value, p)
    );
    exact(&sol, p);
}

#[test]
fn wave_dalembert() {
    let s = P::new();
    let p = &s.pool;
    let c = p.symbol("c", Domain::Positive);
    let f = p.func("f", vec![s.x()]);
    let g = p.func("g", vec![s.x()]);
    let u = s.u();
    let prob = PdeProblem::new(&[s.wave(c)], &[u], p)
        .unwrap()
        .on(s.line(), p)
        .unwrap()
        .ic(s.at(u, s.t(), 0), f, p)
        .unwrap()
        .ic(s.at(s.dx(u, s.t(), 1), s.t(), 0), g, p)
        .unwrap();
    let sol = fourier_method(&prob, p).unwrap();
    assert_eq!(
        sol.kind,
        SolutionKind::ClosedForm,
        "{}",
        shown(sol.value, p)
    );
    exact(&sol, p);
    let v = shown(sol.value, p);
    assert!(v.contains("f(") && v.contains("Integral(g(s)"), "{v}");
}

#[test]
fn schrodinger_kernel() {
    let s = P::new();
    let p = &s.pool;
    let u = s.u();
    let eq = p.add(vec![
        p.mul(vec![p.imaginary_unit(), s.dx(u, s.t(), 1)]),
        s.dx(u, s.x(), 2),
    ]);
    let prob = PdeProblem::new(&[eq], &[u], p)
        .unwrap()
        .on(s.line(), p)
        .unwrap()
        .ic(s.at(u, s.t(), 0), p.func("f", vec![s.x()]), p)
        .unwrap();
    let sol = fourier_method(&prob, p).unwrap();
    assert_eq!(sol.kind, SolutionKind::IntegralRepresentation);
    assert!(
        sol.checks
            .iter()
            .any(|c| c.what == "kernel_pde" && c.status == ZeroStatus::ProvedZero),
        "{:?}",
        sol.checks
    );
}

#[test]
fn laplace_half_plane_poisson_kernel() {
    let s = P::new();
    let p = &s.pool;
    let x = s.x();
    let y = s.t();
    let u = s.u();
    let eq = p.add(vec![s.dx(u, x, 2), s.dx(u, y, 2)]);
    let prob = PdeProblem::new(&[eq], &[u], p)
        .unwrap()
        .on(s.line(), p)
        .unwrap()
        .bc(s.at(u, y, 0), p.func("f", vec![x]), p)
        .unwrap();
    let sol = fourier_method(&prob, p).unwrap();
    assert_eq!(sol.kind, SolutionKind::IntegralRepresentation);
    exact(&sol, p);
    let k = shown(sol.terms[0].kernel, p);
    assert!(!k.contains("exp"), "Poisson kernel is rational: {k}");
}

#[test]
fn klein_gordon_spectral_representation() {
    let s = P::new();
    let p = &s.pool;
    let u = s.u();
    let m = p.symbol("m", Domain::Positive);
    let eq = p.add(vec![
        s.wave(p.integer(1)),
        p.mul(vec![p.pow(m, p.integer(2)), u]),
    ]);
    let prob = PdeProblem::new(&[eq], &[u], p)
        .unwrap()
        .on(s.line(), p)
        .unwrap()
        .ic(s.at(u, s.t(), 0), p.func("f", vec![s.x()]), p)
        .unwrap()
        .ic(s.at(s.dx(u, s.t(), 1), s.t(), 0), p.integer(0), p)
        .unwrap();
    let sol = fourier_method(&prob, p).unwrap();
    assert_eq!(sol.kind, SolutionKind::IntegralRepresentation);
    assert!(
        sol.checks.iter().any(|c| c.what == "integrand_pde"),
        "{:?}",
        sol.checks
    );
    assert_ne!(sol.status, VerificationStatus::Refuted);
}

#[test]
fn heat_half_line_constant_boundary_is_erfc() {
    let s = P::new();
    let p = &s.pool;
    let x = s.x();
    let u = s.u();
    let k = p.symbol("k", Domain::Positive);
    let prob = PdeProblem::new(&[s.heat(k)], &[u], p)
        .unwrap()
        .on(s.quadrant(), p)
        .unwrap()
        .bc(s.at(u, x, 0), p.integer(1), p)
        .unwrap()
        .ic(s.at(u, s.t(), 0), p.integer(0), p)
        .unwrap();
    let sol = transform_solve(&prob, p).unwrap();
    assert_eq!(sol.kind, SolutionKind::ClosedForm);
    assert!(
        shown(sol.value, p).contains("erfc"),
        "{}",
        shown(sol.value, p)
    );
    exact(&sol, p);
}

#[test]
fn heat_half_line_general_boundary_is_duhamel() {
    let s = P::new();
    let p = &s.pool;
    let x = s.x();
    let u = s.u();
    let prob = PdeProblem::new(&[s.heat(p.integer(1))], &[u], p)
        .unwrap()
        .on(s.quadrant(), p)
        .unwrap()
        .bc(s.at(u, x, 0), p.func("g", vec![s.t()]), p)
        .unwrap()
        .ic(s.at(u, s.t(), 0), p.integer(0), p)
        .unwrap();
    let sol = laplace_method(&prob, p).unwrap();
    assert_eq!(sol.kind, SolutionKind::IntegralRepresentation);
    exact(&sol, p);
    assert_eq!(sol.terms[0].upper, s.t());
}

#[test]
fn wave_half_line_signal() {
    let s = P::new();
    let p = &s.pool;
    let x = s.x();
    let u = s.u();
    let c = p.symbol("c", Domain::Positive);
    let prob = PdeProblem::new(&[s.wave(c)], &[u], p)
        .unwrap()
        .on(s.quadrant(), p)
        .unwrap()
        .bc(s.at(u, x, 0), p.func("g", vec![s.t()]), p)
        .unwrap()
        .ic(s.at(u, s.t(), 0), p.integer(0), p)
        .unwrap()
        .ic(s.at(s.dx(u, s.t(), 1), s.t(), 0), p.integer(0), p)
        .unwrap();
    let sol = transform_solve(&prob, p).unwrap();
    assert_eq!(sol.kind, SolutionKind::ClosedForm);
    exact(&sol, p);
    assert!(
        shown(sol.value, p).contains("Heaviside"),
        "{}",
        shown(sol.value, p)
    );
}

#[test]
fn heat_half_line_images() {
    let s = P::new();
    let p = &s.pool;
    let x = s.x();
    let u = s.u();
    for (lhs, name) in [
        (s.at(u, x, 0), "fourier_sine"),
        (s.at(s.dx(u, x, 1), x, 0), "fourier_cosine"),
    ] {
        let prob = PdeProblem::new(&[s.heat(p.integer(1))], &[u], p)
            .unwrap()
            .on(s.quadrant(), p)
            .unwrap()
            .bc(lhs, p.integer(0), p)
            .unwrap()
            .ic(s.at(u, s.t(), 0), p.func("f", vec![x]), p)
            .unwrap();
        let sol = transform_solve(&prob, p).unwrap();
        assert_eq!(sol.method, name);
        assert_eq!(sol.kind, SolutionKind::IntegralRepresentation);
        exact(&sol, p);
    }
}

#[test]
fn superposition_on_the_half_line() {
    let s = P::new();
    let p = &s.pool;
    let x = s.x();
    let u = s.u();
    let prob = PdeProblem::new(&[s.heat(p.integer(1))], &[u], p)
        .unwrap()
        .on(s.quadrant(), p)
        .unwrap()
        .bc(s.at(u, x, 0), p.integer(1), p)
        .unwrap()
        .ic(s.at(u, s.t(), 0), p.func("f", vec![x]), p)
        .unwrap();
    let sol = transform_solve(&prob, p).unwrap();
    assert_eq!(sol.method, "fourier_sine+laplace");
    exact(&sol, p);
}

#[test]
fn refusals() {
    let s = P::new();
    let p = &s.pool;
    let u = s.u();
    // nonlinear
    let eq = p.add(vec![s.dx(u, s.t(), 1), p.mul(vec![u, s.dx(u, s.x(), 1)])]);
    let prob = PdeProblem::new(&[eq], &[u], p)
        .unwrap()
        .on(s.line(), p)
        .unwrap();
    assert_eq!(fourier_method(&prob, p).unwrap_err().code(), "E-PDE-160");
    // no initial data
    let prob = PdeProblem::new(&[s.heat(p.integer(1))], &[u], p)
        .unwrap()
        .on(s.line(), p)
        .unwrap();
    assert_eq!(fourier_method(&prob, p).unwrap_err().code(), "E-PDE-161");
}
