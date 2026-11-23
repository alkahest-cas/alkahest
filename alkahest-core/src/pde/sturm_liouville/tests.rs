use super::*;
use crate::errors::AlkahestError;

fn sh(e: ExprId, pool: &ExprPool) -> String {
    pool.display(e).to_string()
}

fn f64_of(e: ExprId, pool: &ExprPool) -> f64 {
    let b = IntervalEval::new(128).eval(e, pool).expect("numeric");
    b.mid.to_f64()
}

fn mid(enc: &EigenEnclosure) -> f64 {
    (enc.lo.to_f64() + enc.hi.to_f64()) / 2.0
}

struct Ctx {
    pool: ExprPool,
}

impl Ctx {
    fn new() -> Self {
        Ctx {
            pool: ExprPool::new(),
        }
    }
    fn x(&self) -> ExprId {
        self.pool.symbol("x", Domain::Real)
    }
    fn int(&self, k: i64) -> ExprId {
        self.pool.integer(k)
    }
    fn sep(&self, l: EndCondition, r: EndCondition) -> SlBoundary {
        SlBoundary::Separated { left: l, right: r }
    }
}

fn exact(fam: &EigenFamily) {
    assert_eq!(
        fam.verification.status,
        VerificationStatus::ExactlyVerified,
        "{:?}",
        fam.verification
    );
}

#[test]
fn dirichlet_dirichlet() {
    let c = Ctx::new();
    let p = &c.pool;
    let l = p.symbol("L", Domain::Positive);
    let pr = SturmLiouvilleProblem::interval(
        c.x(),
        c.int(0),
        l,
        c.sep(EndCondition::Dirichlet, EndCondition::Dirichlet),
        p,
    )
    .unwrap();
    let sys = sturm_liouville(&pr, p).unwrap();
    assert_eq!(sys.kind, "regular_constant");
    let f = &sys.families[0];
    assert!(f.closed_form);
    exact(f);
    assert_eq!(
        sh(f.eigenvalue, p),
        sh(
            simp(
                p.pow(
                    p.mul(vec![
                        f.index,
                        p.symbol("pi", Domain::Real),
                        p.pow(l, p.integer(-1))
                    ]),
                    p.integer(2)
                ),
                p
            ),
            p
        )
    );
    assert!(
        sh(f.eigenfunctions[0], p).contains("sin"),
        "{}",
        sh(f.eigenfunctions[0], p)
    );
    let norm = f.norms_squared[0].unwrap();
    assert!(
        proved_zero(sub(norm, p.mul(vec![l, p.rational(1, 2)]), p), p),
        "{}",
        sh(norm, p)
    );
    assert!(sys.modes.is_empty());
    assert_eq!(sys.lower, LowerSpectrum::None);
    assert!(sys.side_conditions.is_empty(), "{:?}", sys.side_conditions);
}

#[test]
fn neumann_neumann_has_constant_mode() {
    let c = Ctx::new();
    let p = &c.pool;
    let pr = SturmLiouvilleProblem::interval(
        c.x(),
        c.int(0),
        c.int(2),
        c.sep(EndCondition::Neumann, EndCondition::Neumann),
        p,
    )
    .unwrap();
    let sys = sturm_liouville(&pr, p).unwrap();
    let f = &sys.families[0];
    exact(f);
    assert!(sh(f.eigenfunctions[0], p).contains("cos"));
    assert_eq!(sys.modes.len(), 1);
    assert_eq!(sys.modes[0].eigenvalue, p.integer(0));
    assert_eq!(sys.modes[0].condition, None);
    // (nπ/2)²: λ₁ = π²/4.
    let enc = &f.enclosures;
    assert_eq!(enc.len(), 5);
    assert!((mid(&enc[0]) - std::f64::consts::PI.powi(2) / 4.0).abs() < 1e-12);
}

#[test]
fn dirichlet_neumann_half_integers() {
    let c = Ctx::new();
    let p = &c.pool;
    let pr = SturmLiouvilleProblem::interval(
        c.x(),
        c.int(0),
        c.int(1),
        c.sep(EndCondition::Dirichlet, EndCondition::Neumann),
        p,
    )
    .unwrap();
    let sys = sturm_liouville(&pr, p).unwrap();
    let f = &sys.families[0];
    assert!(f.closed_form, "{}", sh(f.frequency, p));
    exact(f);
    let pi = std::f64::consts::PI;
    assert!((mid(&f.enclosures[0]) - (pi / 2.0).powi(2)).abs() < 1e-12);
    assert!((mid(&f.enclosures[1]) - (1.5 * pi).powi(2)).abs() < 1e-10);
    let norm = f.norms_squared[0].unwrap();
    assert!((f64_of(norm, p) - 0.5).abs() < 1e-14, "{}", sh(norm, p));
}

#[test]
fn robin_family_symbolic_h() {
    let c = Ctx::new();
    let p = &c.pool;
    let l = p.symbol("L", Domain::Positive);
    let h = p.symbol("h", Domain::Positive);
    let pr = SturmLiouvilleProblem::interval(
        c.x(),
        c.int(0),
        l,
        c.sep(
            EndCondition::Dirichlet,
            EndCondition::Robin {
                alpha: h,
                beta: p.integer(1),
            },
        ),
        p,
    )
    .unwrap();
    let sys = sturm_liouville(&pr, p).unwrap();
    let f = &sys.families[0];
    assert!(!f.closed_form);
    exact(f);
    assert_eq!(f.verification.method, "defining_equation");
    let de = sh(f.defining_equation.unwrap(), p);
    assert!(
        de.contains("sin") && de.contains("cos") && !de.contains("tan"),
        "{de}"
    );
    assert!(f.enclosures.is_empty());
    assert!(sys.modes.is_empty(), "hL + 1 > 0: no zero mode");
    assert_eq!(sys.lower, LowerSpectrum::None);
    // Numeric h = 1, L = 1: tan ν = −ν, ν₁ ≈ 2.028757838110434.
    let mut params = HashMap::new();
    params.insert(h, p.integer(1));
    params.insert(l, p.integer(1));
    let enc = f.enclosures(2, &params, 64, p).unwrap();
    assert!(
        (mid(&enc[0]) - 2.028757838110434f64.powi(2)).abs() < 1e-9,
        "{}",
        mid(&enc[0])
    );
    assert!((mid(&enc[1]) - 4.913180439434884f64.powi(2)).abs() < 1e-9);
}

#[test]
fn robin_with_negative_coefficient_has_a_negative_eigenvalue() {
    // X(0) = 0, X'(1) − 2X(1) = 0: tanh κ = κ/2 has a positive root.
    let c = Ctx::new();
    let p = &c.pool;
    let pr = SturmLiouvilleProblem::interval(
        c.x(),
        c.int(0),
        c.int(1),
        c.sep(
            EndCondition::Dirichlet,
            EndCondition::Robin {
                alpha: p.integer(-2),
                beta: p.integer(1),
            },
        ),
        p,
    )
    .unwrap();
    let sys = sturm_liouville(&pr, p).unwrap();
    match &sys.lower {
        LowerSpectrum::Present { family, .. } => {
            let e = &family.enclosures[0];
            // κ ≈ 1.9150080481545, λ = −κ².
            assert!(
                (mid(e) + 1.915008048154_f64.powi(2)).abs() < 1e-6,
                "{}",
                mid(e)
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn periodic_two_eigenfunctions() {
    let c = Ctx::new();
    let p = &c.pool;
    let pi = p.symbol("pi", Domain::Real);
    let pr =
        SturmLiouvilleProblem::interval(c.x(), neg(pi, p), pi, SlBoundary::Periodic, p).unwrap();
    let sys = sturm_liouville(&pr, p).unwrap();
    assert_eq!(sys.kind, "periodic");
    let f = &sys.families[0];
    assert_eq!(f.eigenfunctions.len(), 2);
    exact(f);
    assert_eq!(sh(f.eigenvalue, p), sh(p.pow(f.index, p.integer(2)), p));
    assert_eq!(sys.modes[0].eigenfunction, p.integer(1));
}

#[test]
fn euler_cauchy_closed_form() {
    // (x X')' + λ X / x = 0, X(1) = X(2) = 0: λ_n = (nπ/ln 2)².
    let c = Ctx::new();
    let p = &c.pool;
    let x = c.x();
    let pr = SturmLiouvilleProblem::new(
        x,
        x,
        p.integer(0),
        p.pow(x, p.integer(-1)),
        p.integer(1),
        p.integer(2),
        c.sep(EndCondition::Dirichlet, EndCondition::Dirichlet),
        p,
    )
    .unwrap();
    let sys = sturm_liouville(&pr, p).unwrap();
    assert_eq!(sys.kind, "regular");
    let f = &sys.families[0];
    assert!(f.closed_form, "{}", sh(f.frequency, p));
    exact(f);
    let want = (std::f64::consts::PI / 2f64.ln()).powi(2);
    assert!(
        (mid(&f.enclosures[0]) - want).abs() < 1e-9,
        "{}",
        mid(&f.enclosures[0])
    );
    assert_eq!(sys.lower, LowerSpectrum::None);
}

#[test]
fn bessel_dirichlet_zeros() {
    let c = Ctx::new();
    let p = &c.pool;
    let r = p.symbol("r", Domain::Positive);
    let pr =
        SturmLiouvilleProblem::bessel(r, p.integer(0), p.integer(1), EndCondition::Dirichlet, p)
            .unwrap();
    let sys = sturm_liouville(&pr, p).unwrap();
    assert_eq!(sys.kind, "bessel");
    let f = &sys.families[0];
    exact(f);
    let j01 = 2.404825557695773_f64;
    assert!(
        (mid(&f.enclosures[0]) - j01 * j01).abs() < 1e-9,
        "{:?}",
        f.enclosures
    );
    assert!(sys.modes.is_empty());
    assert_eq!(sys.lower, LowerSpectrum::None);
}

#[test]
fn bessel_order_two_and_neumann_zero_mode() {
    let c = Ctx::new();
    let p = &c.pool;
    let r = p.symbol("r", Domain::Positive);
    let pr =
        SturmLiouvilleProblem::bessel(r, p.integer(2), p.integer(1), EndCondition::Dirichlet, p)
            .unwrap();
    let sys = sturm_liouville(&pr, p).unwrap();
    let j21 = 5.135622301840683_f64;
    assert!((mid(&sys.families[0].enclosures[0]) - j21 * j21).abs() < 1e-8);
    exact(&sys.families[0]);
    let pr = SturmLiouvilleProblem::bessel(r, p.integer(0), p.integer(1), EndCondition::Neumann, p)
        .unwrap();
    let sys = sturm_liouville(&pr, p).unwrap();
    assert_eq!(sys.modes.len(), 1);
    assert_eq!(sys.modes[0].eigenfunction, p.integer(1));
    // J₀'(μ) = −J₁(μ): j₁,₁ ≈ 3.8317059702075125.
    let j11 = 3.831_705_970_207_512_5_f64;
    assert!((mid(&sys.families[0].enclosures[0]) - j11 * j11).abs() < 1e-8);
}

#[test]
fn legendre_hermite_laguerre_chebyshev() {
    let c = Ctx::new();
    let p = &c.pool;
    let x = c.x();
    for (pr, kind, l2) in [
        (
            SturmLiouvilleProblem::legendre(x, p).unwrap(),
            "legendre",
            6.0,
        ),
        (
            SturmLiouvilleProblem::hermite(x, p).unwrap(),
            "hermite",
            4.0,
        ),
        (
            SturmLiouvilleProblem::laguerre(x, p).unwrap(),
            "laguerre",
            2.0,
        ),
        (
            SturmLiouvilleProblem::chebyshev(x, p).unwrap(),
            "chebyshev",
            9.0,
        ),
    ] {
        let sys = sturm_liouville(&pr, p).unwrap();
        assert_eq!(sys.kind, kind);
        let f = &sys.families[0];
        exact(f);
        // the third enclosure: λ₂ (λ₃ for Chebyshev, which starts at 1)
        assert!(
            (mid(&f.enclosures[2]) - l2).abs() < 1e-12,
            "{kind}: {:?}",
            f.enclosures
        );
        let fs = f.eigenfunctions_at(3, p).unwrap();
        assert!(!sh(fs[0], p).contains("_"), "{kind}: {}", sh(fs[0], p));
    }
}

#[test]
fn from_ode_reads_the_coefficients() {
    let c = Ctx::new();
    let p = &c.pool;
    let x = c.x();
    let lam = p.symbol("lambda", Domain::Real);
    let xf = p.func("X", vec![x]);
    let d2 = crate::diff::diff(crate::diff::diff(xf, x, p).unwrap().value, x, p)
        .unwrap()
        .value;
    let eq = p.add(vec![d2, p.mul(vec![lam, xf])]);
    let pr = SturmLiouvilleProblem::from_ode(
        eq,
        xf,
        lam,
        c.int(0),
        p.symbol("pi", Domain::Real),
        c.sep(EndCondition::Dirichlet, EndCondition::Dirichlet),
        p,
    )
    .unwrap();
    assert_eq!(pr.p, p.integer(1));
    assert_eq!(pr.w, p.integer(1));
    let sys = sturm_liouville(&pr, p).unwrap();
    assert_eq!(
        sh(sys.families[0].eigenvalue, p),
        sh(p.pow(sys.families[0].index, p.integer(2)), p)
    );
    // Legendre written out: (1 − x²) X'' − 2x X' + λ X = 0.
    let d1 = crate::diff::diff(xf, x, p).unwrap().value;
    let eq = p.add(vec![
        p.mul(vec![sub(p.integer(1), p.pow(x, p.integer(2)), p), d2]),
        p.mul(vec![p.integer(-2), x, d1]),
        p.mul(vec![lam, xf]),
    ]);
    let pr = SturmLiouvilleProblem::from_ode(
        eq,
        xf,
        lam,
        p.integer(-1),
        p.integer(1),
        c.sep(EndCondition::Bounded, EndCondition::Bounded),
        p,
    )
    .unwrap();
    assert_eq!(sturm_liouville(&pr, p).unwrap().kind, "legendre");
}

#[test]
fn refusals() {
    let c = Ctx::new();
    let p = &c.pool;
    let bad = SturmLiouvilleProblem::interval(
        c.x(),
        c.int(0),
        c.int(1),
        c.sep(
            EndCondition::Robin {
                alpha: p.integer(0),
                beta: p.integer(0),
            },
            EndCondition::Dirichlet,
        ),
        p,
    );
    assert_eq!(bad.unwrap_err().code(), "E-PDE-151");
    let pr = SturmLiouvilleProblem::interval(
        c.x(),
        c.int(0),
        c.int(1),
        c.sep(EndCondition::Bounded, EndCondition::Dirichlet),
        p,
    )
    .unwrap();
    assert_eq!(sturm_liouville(&pr, p).unwrap_err().code(), "E-PDE-151");
}

#[test]
fn error_codes_in_m3_range() {
    for e in [
        PdeError::SlUnsupported(String::new()),
        PdeError::SlInvalidBoundary(String::new()),
        PdeError::SlDegenerate(String::new()),
        PdeError::SlVerificationFailed(String::new()),
        PdeError::TransformMethodUnsupported(String::new()),
        PdeError::TransformData(String::new()),
        PdeError::TransformOde(String::new()),
        PdeError::TransformVerificationFailed(String::new()),
    ] {
        let n: u32 = e.code()[6..].parse().unwrap();
        assert!((150..=199).contains(&n));
        assert!(e.remediation().is_some(), "{}", e.code());
    }
}
