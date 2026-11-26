use super::*;
use crate::errors::AlkahestError;
use crate::kernel::Domain;

struct Ctx {
    pool: ExprPool,
}

impl Ctx {
    fn new() -> Self {
        Ctx {
            pool: ExprPool::new(),
        }
    }
    fn sym(&self, n: &str) -> ExprId {
        self.pool.symbol(n, Domain::Real)
    }
    fn d(&self, e: ExprId, v: ExprId) -> ExprId {
        crate::diff::diff(e, v, &self.pool).unwrap().value
    }
    fn int(&self, n: i64) -> ExprId {
        self.pool.integer(n)
    }
    /// Σ cᵢ·tᵢ
    fn lin(&self, terms: &[(ExprId, ExprId)]) -> ExprId {
        self.pool.add(
            terms
                .iter()
                .map(|&(c, t)| self.pool.mul(vec![c, t]))
                .collect(),
        )
    }
    fn solve(&self, eq: ExprId, u: ExprId) -> Result<SecondOrderSolution, PdeError> {
        let p = PdeProblem::new(&[eq], &[u], &self.pool).unwrap();
        solve_second_order(&p, &self.pool)
    }
}

fn xy(c: &Ctx) -> (ExprId, ExprId, ExprId) {
    let x = c.sym("x");
    let y = c.sym("y");
    (x, y, c.pool.func("u", vec![x, y]))
}

#[test]
fn wave_general_solution_is_dalembert() {
    let c = Ctx::new();
    let x = c.sym("x");
    let t = c.sym("t");
    let s = c.sym("c");
    let u = c.pool.func("u", vec![x, t]);
    let p = &c.pool;
    let eq = c.lin(&[
        (c.int(1), c.d(c.d(u, t), t)),
        (
            p.mul(vec![c.int(-1), p.pow(s, c.int(2))]),
            c.d(c.d(u, x), x),
        ),
    ]);
    let r = c.solve(eq, u).unwrap();
    assert_eq!(r.kind, "general");
    assert!(r.verification.status.is_verified());
    assert_eq!(r.arbitrary_functions.len(), 2, "{}", p.display(r.solution));
}

#[test]
fn laplace_equation_complex_characteristics() {
    let c = Ctx::new();
    let (x, y, u) = xy(&c);
    let eq = c.pool.add(vec![c.d(c.d(u, x), x), c.d(c.d(u, y), y)]);
    let r = c.solve(eq, u).unwrap();
    assert_eq!(r.kind, "general");
    assert_eq!(r.arbitrary_functions.len(), 2);
    let cf = canonical_form(eq, u, &c.pool).unwrap();
    assert_eq!(cf.pde_type, PdeType::Elliptic);
}

#[test]
fn rational_factoring_with_lower_order_terms() {
    // u_xx − 3u_xy + 2u_yy + u_x − u_y = 0 : symbol (ξ − η)(ξ − 2η) + (ξ − η)
    // = (ξ − η)(ξ − 2η + 1)
    let c = Ctx::new();
    let (x, y, u) = xy(&c);
    let eq = c.lin(&[
        (c.int(1), c.d(c.d(u, x), x)),
        (c.int(-3), c.d(c.d(u, x), y)),
        (c.int(2), c.d(c.d(u, y), y)),
        (c.int(1), c.d(u, x)),
        (c.int(-1), c.d(u, y)),
    ]);
    let r = c.solve(eq, u).unwrap();
    assert_eq!(r.kind, "general");
    assert!(
        r.method.contains("factored") || r.method.contains("shift"),
        "{}",
        r.method
    );
}

#[test]
fn repeated_factor_parabolic() {
    // u_xx + 2u_xy + u_yy = 0 → F(x − y) + x G(x − y)
    let c = Ctx::new();
    let (x, y, u) = xy(&c);
    let eq = c.lin(&[
        (c.int(1), c.d(c.d(u, x), x)),
        (c.int(2), c.d(c.d(u, x), y)),
        (c.int(1), c.d(c.d(u, y), y)),
    ]);
    let r = c.solve(eq, u).unwrap();
    assert_eq!(r.kind, "general");
    assert_eq!(r.arbitrary_functions.len(), 2);
}

#[test]
fn exponential_shift_removes_first_order_terms() {
    // u_xx − u_yy + 2u_x + u = 0  →  u = e^{−x}(F(x+y) + G(x−y))
    let c = Ctx::new();
    let (x, y, u) = xy(&c);
    let eq = c.lin(&[
        (c.int(1), c.d(c.d(u, x), x)),
        (c.int(-1), c.d(c.d(u, y), y)),
        (c.int(2), c.d(u, x)),
        (c.int(1), u),
    ]);
    let r = c.solve(eq, u).unwrap();
    assert_eq!(r.kind, "general");
    assert_eq!(r.method, "exponential_shift");
}

#[test]
fn inhomogeneous_wave_particular_by_characteristics() {
    let c = Ctx::new();
    let (x, y, u) = xy(&c);
    let p = &c.pool;
    let eq = c.lin(&[
        (c.int(1), c.d(c.d(u, x), x)),
        (c.int(-1), c.d(c.d(u, y), y)),
        (c.int(-1), p.mul(vec![x, y])),
    ]);
    let r = c.solve(eq, u).unwrap();
    assert_eq!(r.kind, "general");
}

#[test]
fn ode_in_one_variable() {
    let c = Ctx::new();
    let (x, y, u) = xy(&c);
    let eq = c.pool.add(vec![c.d(c.d(u, x), x), u]);
    let r = c.solve(eq, u).unwrap();
    assert_eq!(r.method, "ode_in_one_variable");
    // constants became functions of y
    let _ = y;
    assert_eq!(r.arbitrary_functions.len(), 2);
}

#[test]
fn heat_gives_a_labelled_particular_family() {
    let c = Ctx::new();
    let x = c.sym("x");
    let t = c.sym("t");
    let u = c.pool.func("u", vec![x, t]);
    let eq = c.lin(&[(c.int(1), c.d(u, t)), (c.int(-1), c.d(c.d(u, x), x))]);
    let r = c.solve(eq, u).unwrap();
    assert_eq!(r.kind, "particular");
    assert_eq!(r.method, "dispersion_relation");
    assert_eq!(r.parameters.len(), 1);
}

#[test]
fn dalembert_cauchy_problem() {
    let c = Ctx::new();
    let x = c.sym("x");
    let t = c.sym("t");
    let s = c.pool.symbol("c", Domain::Positive);
    let u = c.pool.func("u", vec![x, t]);
    let p = &c.pool;
    let eq = c.lin(&[
        (c.int(1), c.d(c.d(u, t), t)),
        (
            p.mul(vec![c.int(-1), p.pow(s, c.int(2))]),
            c.d(c.d(u, x), x),
        ),
    ]);
    let zero = c.int(0);
    let u0 = p.func("u", vec![x, zero]);
    let ut0 = crate::diff::undefined::derivative_of(u0, &[0, 1], p).unwrap();
    let prob = PdeProblem::new(&[eq], &[u], p)
        .unwrap()
        .ic(u0, p.func("sin", vec![x]), p)
        .unwrap()
        .ic(ut0, p.func("cos", vec![x]), p)
        .unwrap();
    let r = solve_second_order(&prob, p).unwrap();
    assert_eq!(r.method, "dalembert");
    assert_eq!(r.kind, "particular");
    assert!(r.verification.status.is_verified());
    assert_eq!(r.verification.checks.len(), 3);
}

#[test]
fn dalembert_with_arbitrary_displacement() {
    // u(x,0) = φ(x) (an undefined function), u_t(x,0) = 0
    let c = Ctx::new();
    let x = c.sym("x");
    let t = c.sym("t");
    let u = c.pool.func("u", vec![x, t]);
    let p = &c.pool;
    let eq = c.lin(&[
        (c.int(1), c.d(c.d(u, t), t)),
        (c.int(-4), c.d(c.d(u, x), x)),
    ]);
    let zero = c.int(0);
    let u0 = p.func("u", vec![x, zero]);
    let ut0 = crate::diff::undefined::derivative_of(u0, &[0, 1], p).unwrap();
    let prob = PdeProblem::new(&[eq], &[u], p)
        .unwrap()
        .ic(u0, p.func("phi", vec![x]), p)
        .unwrap()
        .ic(ut0, zero, p)
        .unwrap();
    let r = solve_second_order(&prob, p).unwrap();
    assert!(r.verification.status.is_verified());
}

#[test]
fn laplace_cascade_epd_terminates() {
    // Euler–Poisson–Darboux u_xy + (N/(x+y))(u_x + u_y) = 0 for N = 1, 2.
    for n in [1i64, 2] {
        let c = Ctx::new();
        let (x, y, u) = xy(&c);
        let p = &c.pool;
        let coef = p.mul(vec![c.int(n), p.pow(p.add(vec![x, y]), c.int(-1))]);
        let eq = c.lin(&[
            (c.int(1), c.d(c.d(u, x), y)),
            (coef, c.d(u, x)),
            (coef, c.d(u, y)),
        ]);
        let r = laplace_cascade(eq, u, 8, p).unwrap();
        assert_eq!(r.kind, "general");
        assert!(r.verification.status.is_verified());
        let r2 = c.solve(eq, u).unwrap();
        assert!(r2.method.starts_with("laplace_cascade"));
    }
}

#[test]
fn laplace_cascade_budget_refuses_klein_gordon() {
    let c = Ctx::new();
    let (x, y, u) = xy(&c);
    let eq = c.lin(&[(c.int(1), c.d(c.d(u, x), y)), (c.int(-1), u)]);
    let e = laplace_cascade(eq, u, 4, &c.pool).unwrap_err();
    assert_eq!(e.code(), "E-PDE-103");
    // solve_second_order still offers the exponential family
    let r = c.solve(eq, u).unwrap();
    assert_eq!(r.kind, "particular");
}

#[test]
fn variable_coefficient_parabolic_via_canonical_form() {
    // x² u_xx + 2xy u_xy + y² u_yy = 0 → F(y/x) + x G(y/x)
    let c = Ctx::new();
    let (x, y, u) = xy(&c);
    let p = &c.pool;
    let eq = c.lin(&[
        (p.pow(x, c.int(2)), c.d(c.d(u, x), x)),
        (p.mul(vec![c.int(2), x, y]), c.d(c.d(u, x), y)),
        (p.pow(y, c.int(2)), c.d(c.d(u, y), y)),
    ]);
    let cf = canonical_form(eq, u, p).unwrap();
    assert_eq!(cf.pde_type, PdeType::Parabolic);
    let r = c.solve(eq, u).unwrap();
    assert_eq!(r.kind, "general");
    assert!(r.method.starts_with("canonical_form"), "{}", r.method);
}

#[test]
fn errors_are_in_the_m2_range() {
    let all = [
        PdeError::NotSecondOrderLinear2D(String::new()),
        PdeError::CharacteristicsNotIntegrated(String::new()),
        PdeError::TypeNotUniform(String::new()),
        PdeError::LaplaceBudgetExhausted(String::new()),
        PdeError::NoClosedForm(String::new()),
        PdeError::SecondOrderNotVerified(String::new()),
        PdeError::QuadratureOfArbitraryFunction(String::new()),
        PdeError::CauchyDataUnsupported(String::new()),
        PdeError::NotLaplaceForm(String::new()),
    ];
    for e in all {
        let code = e.code();
        let n: u32 = code.strip_prefix("E-PDE-").unwrap().parse().unwrap();
        assert!((100..=149).contains(&n), "{code}");
        assert!(e.remediation().is_some(), "{code} not registered");
    }
}

#[test]
fn not_second_order_is_refused() {
    let c = Ctx::new();
    let (x, _y, u) = xy(&c);
    let eq = c.d(u, x);
    assert_eq!(c.solve(eq, u).unwrap_err().code(), "E-PDE-100");
}

#[test]
fn tricomi_has_one_form_per_region() {
    let c = Ctx::new();
    let (x, y, u) = xy(&c);
    let p = &c.pool;
    let eq = c.lin(&[(y, c.d(c.d(u, x), x)), (c.int(1), c.d(c.d(u, y), y))]);
    assert_eq!(canonical_form(eq, u, p).unwrap_err().code(), "E-PDE-102");
    let forms = canonical_forms(eq, u, p).unwrap();
    let mut types: Vec<&str> = forms.iter().map(|f| f.pde_type.as_str()).collect();
    types.sort();
    assert_eq!(types, ["elliptic", "hyperbolic"]);
    assert!(forms.iter().all(|f| f.region.is_some()));
}

#[test]
fn cascade_three_steps_verifies() {
    let c = Ctx::new();
    let (x, y, u) = xy(&c);
    let p = &c.pool;
    let coef = p.mul(vec![c.int(-12), p.pow(p.add(vec![x, y]), c.int(-2))]);
    let eq = c.lin(&[(c.int(1), c.d(c.d(u, x), y)), (coef, u)]);
    let r = laplace_cascade(eq, u, 8, p).unwrap();
    assert_eq!(r.method, "laplace_cascade(3 step(s))");
    assert_eq!(
        laplace_cascade(eq, u, 2, p).unwrap_err().code(),
        "E-PDE-103"
    );
}

#[test]
fn canonical_form_is_an_identity() {
    // u(x, y) = w(φ, ψ) substituted into the operator equals the multiplier
    // times the canonical operator at (φ, ψ).
    let c = Ctx::new();
    let (x, y, u) = xy(&c);
    let p = &c.pool;
    let eq = c.lin(&[
        (p.pow(x, c.int(2)), c.d(c.d(u, x), x)),
        (
            p.mul(vec![c.int(-1), p.pow(y, c.int(2))]),
            c.d(c.d(u, y), y),
        ),
    ]);
    for cf in canonical_forms(eq, u, p).unwrap() {
        let (phi, psi) = cf.transformation;
        let (xi, eta) = cf.new_variables;
        let wname = crate::diff::undefined::view(cf.unknown, p).unwrap().name;
        let composed = p.func(wname, vec![phi, psi]);
        let lhs = crate::diff::undefined::replace_function(eq, u, composed, p).unwrap();
        let rhs = p.mul(vec![
            cf.multiplier,
            util::subs2(cf.equation, (xi, phi), (eta, psi), p),
        ]);
        assert!(
            util::is_zero(util::sub(lhs, rhs, p), p),
            "{}",
            p.display(cf.equation)
        );
    }
}
