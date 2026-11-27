use super::*;
use crate::kernel::{Domain, ExprPool};

fn table(op: impl Fn(&ExprPool) -> GreenOperator, dims: &[usize], timed: bool) {
    let p = ExprPool::new();
    let names = ["x", "y", "z"];
    for &n in dims {
        let space: Vec<_> = names[..n]
            .iter()
            .map(|s| p.symbol(*s, Domain::Real))
            .collect();
        let t = timed.then(|| p.symbol("t", Domain::Positive));
        let fs = fundamental_solution(op(&p), &space, t, &p).unwrap();
        for c in &fs.checks {
            eprintln!(
                "{} n={n} {} {} {} [{}]",
                fs.operator.name(),
                c.what,
                c.status.as_str(),
                c.method,
                c.detail
            );
        }
        eprintln!("  G = {}", p.display(fs.kernel));
        assert_eq!(
            fs.status,
            VerificationStatus::ExactlyVerified,
            "{} n={n}",
            fs.operator.name()
        );
    }
}

#[test]
fn laplace_table() {
    table(|_| GreenOperator::Laplace, &[1, 2, 3], false);
}

#[test]
fn biharmonic_table() {
    table(|_| GreenOperator::Biharmonic, &[1, 2, 3], false);
}

#[test]
fn helmholtz_table() {
    table(
        |p| GreenOperator::Helmholtz {
            k: p.symbol("k", Domain::Positive),
        },
        &[1, 2, 3],
        false,
    );
}

#[test]
fn modified_helmholtz_table() {
    table(
        |p| GreenOperator::ModifiedHelmholtz {
            k: p.symbol("k", Domain::Positive),
        },
        &[1, 2, 3],
        false,
    );
}

#[test]
fn heat_table() {
    table(
        |p| GreenOperator::Heat {
            diffusivity: p.symbol("kappa", Domain::Positive),
        },
        &[1, 2, 3],
        true,
    );
}

#[test]
fn wave_table() {
    table(
        |p| GreenOperator::Wave {
            speed: p.symbol("c", Domain::Positive),
        },
        &[1, 2, 3],
        true,
    );
}

#[test]
fn klein_gordon_and_telegraph() {
    table(
        |p| GreenOperator::KleinGordon {
            speed: p.symbol("c", Domain::Positive),
            mass: p.symbol("m", Domain::Positive),
        },
        &[1],
        true,
    );
    table(
        |p| GreenOperator::Telegraph {
            speed: p.symbol("c", Domain::Positive),
            damping: p.symbol("a", Domain::Positive),
        },
        &[1],
        true,
    );
}

#[test]
fn symbolic_dimension() {
    let p = ExprPool::new();
    let n = p.symbol("n", Domain::Positive);
    let r = p.symbol("r", Domain::Positive);
    let t = p.symbol("t", Domain::Positive);
    for (op, tm) in [
        (GreenOperator::Laplace, None),
        (
            GreenOperator::Heat {
                diffusivity: p.symbol("kappa", Domain::Positive),
            },
            Some(t),
        ),
    ] {
        let fs = fundamental_solution_nd(op, n, r, tm, &p).unwrap();
        for c in &fs.checks {
            eprintln!(
                "{} {} {} {}",
                fs.operator.name(),
                c.what,
                c.status.as_str(),
                c.method
            );
        }
        assert_eq!(
            fs.status,
            VerificationStatus::ExactlyVerified,
            "{}",
            fs.operator.name()
        );
    }
}

fn show(g: &GreenFunction, p: &ExprPool) {
    for c in &g.checks {
        eprintln!(
            "{} {} {} {} {} [{}]",
            g.operator,
            g.method,
            c.what,
            c.status.as_str(),
            c.method,
            c.detail
        );
    }
    eprintln!("  G = {}", p.display(g.value));
    if let Some(pk) = g.poisson_kernel {
        eprintln!("  P = {}", p.display(pk));
    }
}

type ImageCase = (
    GreenOperator,
    Vec<crate::kernel::ExprId>,
    Vec<crate::kernel::ExprId>,
    Option<crate::kernel::ExprId>,
    ImageRegion,
    Vec<WallCondition>,
);

#[test]
fn images_half_and_quarter_space() {
    let p = ExprPool::new();
    let (x, y, z) = (
        p.symbol("x", Domain::Real),
        p.symbol("y", Domain::Real),
        p.symbol("z", Domain::Real),
    );
    let (xi, eta, zeta) = (
        p.symbol("xi", Domain::Real),
        p.symbol("eta", Domain::Real),
        p.symbol("zeta", Domain::Real),
    );
    let t = p.symbol("t", Domain::Positive);
    let k = p.symbol("k", Domain::Positive);
    let kappa = p.symbol("kappa", Domain::Positive);
    let c = p.symbol("c", Domain::Positive);
    use WallCondition::{Dirichlet as Dw, Neumann as Nw};
    let half = |a| ImageRegion::HalfSpace { axis: a };
    let quarter = ImageRegion::QuarterSpace { axes: (0, 1) };
    let heat = GreenOperator::Heat { diffusivity: kappa };
    let cases: Vec<ImageCase> = vec![
        (
            GreenOperator::Laplace,
            vec![x, y],
            vec![xi, eta],
            None,
            half(1),
            vec![Dw],
        ),
        (
            GreenOperator::Laplace,
            vec![x, y],
            vec![xi, eta],
            None,
            half(1),
            vec![Nw],
        ),
        (
            GreenOperator::Laplace,
            vec![x, y, z],
            vec![xi, eta, zeta],
            None,
            half(2),
            vec![Dw],
        ),
        (
            GreenOperator::Laplace,
            vec![x, y],
            vec![xi, eta],
            None,
            quarter,
            vec![Dw, Nw],
        ),
        (heat, vec![x], vec![xi], Some(t), half(0), vec![Dw]),
        (
            heat,
            vec![x, y],
            vec![xi, eta],
            Some(t),
            quarter,
            vec![Nw, Dw],
        ),
        (
            GreenOperator::ModifiedHelmholtz { k },
            vec![x, y, z],
            vec![xi, eta, zeta],
            None,
            half(0),
            vec![Nw],
        ),
        (
            GreenOperator::Helmholtz { k },
            vec![x, y],
            vec![xi, eta],
            None,
            half(1),
            vec![Dw],
        ),
        (
            GreenOperator::Wave { speed: c },
            vec![x],
            vec![xi],
            Some(t),
            half(0),
            vec![Dw],
        ),
        (
            GreenOperator::Wave { speed: c },
            vec![x, y, z],
            vec![xi, eta, zeta],
            Some(t),
            half(0),
            vec![Nw],
        ),
    ];
    for (op, sp, so, tm, reg, w) in cases {
        let g = image_green_function(op, &sp, tm, &so, reg, &w, &p).unwrap();
        show(&g, &p);
        assert_eq!(
            g.status,
            VerificationStatus::ExactlyVerified,
            "{} {:?}",
            g.operator,
            reg
        );
    }
}

#[test]
fn images_strip_and_kelvin() {
    let p = ExprPool::new();
    let (x, y, z) = (
        p.symbol("x", Domain::Real),
        p.symbol("y", Domain::Real),
        p.symbol("z", Domain::Real),
    );
    let (xi, eta, zeta) = (
        p.symbol("xi", Domain::Real),
        p.symbol("eta", Domain::Real),
        p.symbol("zeta", Domain::Real),
    );
    let t = p.symbol("t", Domain::Positive);
    let l = p.symbol("L", Domain::Positive);
    let r = p.symbol("R", Domain::Positive);
    let kappa = p.symbol("kappa", Domain::Positive);
    use WallCondition::{Dirichlet as Dw, Neumann as Nw};
    for w in [[Dw, Dw], [Nw, Nw], [Dw, Nw], [Nw, Dw]] {
        let g = image_green_function(
            GreenOperator::Heat { diffusivity: kappa },
            &[x],
            Some(t),
            &[xi],
            ImageRegion::Strip { axis: 0, width: l },
            &w,
            &p,
        )
        .unwrap();
        show(&g, &p);
        assert_eq!(g.status, VerificationStatus::ExactlyVerified, "heat {w:?}");
    }
    let g = image_green_function(
        GreenOperator::Laplace,
        &[x, y],
        None,
        &[xi, eta],
        ImageRegion::Strip { axis: 1, width: l },
        &[Dw, Dw],
        &p,
    )
    .unwrap();
    show(&g, &p);
    assert_eq!(
        g.status,
        VerificationStatus::ExactlyVerified,
        "laplace strip"
    );
    let g = image_green_function(
        GreenOperator::Laplace,
        &[x, y, z],
        None,
        &[xi, eta, zeta],
        ImageRegion::Strip { axis: 2, width: l },
        &[Dw, Dw],
        &p,
    )
    .unwrap();
    show(&g, &p);
    assert_eq!(
        g.status,
        VerificationStatus::ExactlyVerified,
        "laplace slab"
    );
    for (sp, so, reg) in [
        (vec![x, y], vec![xi, eta], ImageRegion::Disk { radius: r }),
        (
            vec![x, y, z],
            vec![xi, eta, zeta],
            ImageRegion::Ball { radius: r },
        ),
    ] {
        let g =
            image_green_function(GreenOperator::Laplace, &sp, None, &so, reg, &[Dw], &p).unwrap();
        show(&g, &p);
        assert_eq!(g.status, VerificationStatus::ExactlyVerified, "{reg:?}");
    }
    let e = image_green_function(
        GreenOperator::Laplace,
        &[x, y],
        None,
        &[xi, eta],
        ImageRegion::Disk { radius: r },
        &[Nw],
        &p,
    )
    .unwrap_err();
    assert_eq!(crate::errors::AlkahestError::code(&e), "E-PDE-252");
}

#[allow(clippy::too_many_arguments)]
fn sl(
    p: &ExprPool,
    x: crate::kernel::ExprId,
    pp: crate::kernel::ExprId,
    q: crate::kernel::ExprId,
    a: crate::kernel::ExprId,
    b: crate::kernel::ExprId,
    left: crate::pde::EndCondition,
    right: crate::pde::EndCondition,
) -> crate::pde::SturmLiouvilleProblem {
    crate::pde::SturmLiouvilleProblem::new(
        x,
        pp,
        q,
        p.integer(1),
        a,
        b,
        crate::pde::SlBoundary::Separated { left, right },
        p,
    )
    .unwrap()
}

#[test]
fn ode_green_functions() {
    use crate::pde::EndCondition::{Dirichlet, Neumann, Robin};
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Real);
    let xi = p.symbol("xi", Domain::Real);
    let l = p.symbol("L", Domain::Positive);
    let k = p.symbol("k", Domain::Positive);
    let h = p.symbol("h", Domain::Positive);
    let (one, zero) = (p.integer(1), p.integer(0));
    let mk2 = p.mul(vec![p.integer(-1), p.pow(k, p.integer(2))]);
    let cases = vec![
        sl(&p, x, one, zero, zero, l, Dirichlet, Dirichlet),
        sl(&p, x, one, zero, zero, l, Dirichlet, Neumann),
        sl(
            &p,
            x,
            one,
            zero,
            zero,
            l,
            Dirichlet,
            Robin {
                alpha: h,
                beta: one,
            },
        ),
        sl(&p, x, one, mk2, zero, l, Dirichlet, Dirichlet),
        sl(
            &p,
            x,
            one,
            p.pow(k, p.integer(2)),
            zero,
            one,
            Dirichlet,
            Dirichlet,
        ),
    ];
    for prob in cases {
        let g = ode_green_function(&prob, xi, &p).unwrap();
        show(&g, &p);
        assert_eq!(g.status, VerificationStatus::ExactlyVerified);
    }
    // −u'' with Neumann ends: no Green's function.
    let prob = sl(&p, x, one, zero, zero, l, Neumann, Neumann);
    let e = ode_green_function(&prob, xi, &p).unwrap_err();
    assert_eq!(crate::errors::AlkahestError::code(&e), "E-PDE-254");
}

#[test]
fn eigen_series_green_functions() {
    use crate::pde::EndCondition::{Dirichlet, Neumann};
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Real);
    let xi = p.symbol("xi", Domain::Real);
    let l = p.symbol("L", Domain::Positive);
    let t = p.symbol("t", Domain::Positive);
    let kappa = p.symbol("kappa", Domain::Positive);
    let c = p.symbol("c", Domain::Positive);
    let (one, zero) = (p.integer(1), p.integer(0));
    let dd_prob = sl(&p, x, one, zero, zero, l, Dirichlet, Dirichlet);
    for kind in [
        SeriesKind::Static,
        SeriesKind::Heat {
            diffusivity: kappa,
            time: t,
        },
        SeriesKind::Wave { speed: c, time: t },
    ] {
        let g = eigen_green_function(&dd_prob, xi, kind, &p).unwrap();
        show(&g, &p);
        assert_eq!(g.status, VerificationStatus::ExactlyVerified, "{kind:?}");
        if kind == SeriesKind::Static {
            assert!(g.checks.iter().any(|c| c.what == "coefficients"));
        }
    }
    let nn = sl(&p, x, one, zero, zero, l, Neumann, Neumann);
    let g = eigen_green_function(&nn, xi, SeriesKind::Static, &p).unwrap();
    show(&g, &p);
    assert_eq!(g.method, "modified_eigenfunction_series");
    let g = eigen_green_function(
        &nn,
        xi,
        SeriesKind::Heat {
            diffusivity: kappa,
            time: t,
        },
        &p,
    )
    .unwrap();
    show(&g, &p);
    assert_eq!(g.status, VerificationStatus::ExactlyVerified);
}

fn show_sol(s: &GreenSolution, p: &ExprPool) {
    for c in &s.checks {
        eprintln!(
            "{} {} {} {} [{}]",
            s.method,
            c.what,
            c.status.as_str(),
            c.method,
            c.detail
        );
    }
    eprintln!("  u = {} ({})", p.display(s.value), s.kind.as_str());
}

#[test]
fn green_solve_problems() {
    use crate::pde::{PdeDomain, PdeProblem};
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Real);
    let t = p.symbol("t", Domain::Real);
    let l = p.symbol("L", Domain::Positive);
    let c = p.symbol("c", Domain::Positive);
    let u = p.func("u", vec![x]);
    let d1 = crate::diff::undefined::build(u, &[1], &p);
    let d2 = crate::diff::undefined::build(u, &[2], &p);
    let at = |e: crate::kernel::ExprId, v| crate::pde::green::at(e, x, v, &p);
    let (zero, one) = (p.integer(0), p.integer(1));
    // −u'' = 1, u(0) = u(1) = 0  →  x(1 − x)/2
    let eq = p.add(vec![p.mul(vec![p.integer(-1), d2]), p.integer(-1)]);
    let prob = PdeProblem::new(&[eq], &[u], &p)
        .unwrap()
        .on(PdeDomain::interval(x, zero, one, &p).unwrap(), &p)
        .unwrap()
        .bc(at(u, zero), zero, &p)
        .unwrap()
        .bc(at(u, one), zero, &p)
        .unwrap();
    let s = green_solve(&prob, &p).unwrap();
    show_sol(&s, &p);
    assert_eq!(s.kind, GreenSolutionKind::ClosedForm);
    assert_eq!(s.status, VerificationStatus::ExactlyVerified);
    let want = p.mul(vec![
        p.rational(1, 2),
        x,
        p.add(vec![one, p.mul(vec![p.integer(-1), x])]),
    ]);
    assert_eq!(
        zstatus(super::sub(s.value, want, &p), &p),
        crate::simplify::zero::ZeroStatus::ProvedZero
    );
    // u'' = f(x) (symbolic), u(0) = 2, u'(L) = 3: verified representation.
    let f = p.func("f", vec![x]);
    let eq = p.add(vec![d2, p.mul(vec![p.integer(-1), f])]);
    let prob = PdeProblem::new(&[eq], &[u], &p)
        .unwrap()
        .on(PdeDomain::interval(x, zero, l, &p).unwrap(), &p)
        .unwrap()
        .bc(at(u, zero), p.integer(2), &p)
        .unwrap()
        .bc(at(d1, l), p.integer(3), &p)
        .unwrap();
    let s = green_solve(&prob, &p).unwrap();
    show_sol(&s, &p);
    assert_eq!(s.kind, GreenSolutionKind::IntegralRepresentation);
    assert_eq!(s.status, VerificationStatus::ExactlyVerified);
    // u'' + u = x, u(0) = 0, u(1) = 1 (closed).
    let eq = p.add(vec![d2, u, p.mul(vec![p.integer(-1), x])]);
    let prob = PdeProblem::new(&[eq], &[u], &p)
        .unwrap()
        .on(PdeDomain::interval(x, zero, one, &p).unwrap(), &p)
        .unwrap()
        .bc(at(u, zero), zero, &p)
        .unwrap()
        .bc(at(u, one), one, &p)
        .unwrap();
    let s = green_solve(&prob, &p).unwrap();
    show_sol(&s, &p);
    assert_eq!(s.status, VerificationStatus::ExactlyVerified);
    // Wave with a source: u_tt − c² u_xx = x, zero data → x t²/2.
    let w = p.func("w", vec![x, t]);
    let wtt = crate::diff::undefined::build(w, &[0, 2], &p);
    let wxx = crate::diff::undefined::build(w, &[2, 0], &p);
    let eq = p.add(vec![
        wtt,
        p.mul(vec![p.integer(-1), p.pow(c, p.integer(2)), wxx]),
        p.mul(vec![p.integer(-1), x]),
    ]);
    let dom = PdeDomain::product(
        vec![
            PdeDomain::line(x, &p).unwrap(),
            PdeDomain::half_line(t, zero, false, &p).unwrap(),
        ],
        &p,
    )
    .unwrap();
    let w0 = p.func("w", vec![x, zero]);
    let wt0 = crate::diff::undefined::build(w0, &[0, 1], &p);
    let prob = PdeProblem::new(&[eq], &[w], &p)
        .unwrap()
        .on(dom.clone(), &p)
        .unwrap()
        .ic(w0, zero, &p)
        .unwrap()
        .ic(wt0, zero, &p)
        .unwrap();
    let s = green_solve(&prob, &p).unwrap();
    show_sol(&s, &p);
    assert_eq!(s.kind, GreenSolutionKind::ClosedForm);
    assert_eq!(s.status, VerificationStatus::ExactlyVerified);
    // Heat with a source: u_t − u_xx = 1, u(x, 0) = 0 → t.
    let wt = crate::diff::undefined::build(w, &[0, 1], &p);
    let eq = p.add(vec![wt, p.mul(vec![p.integer(-1), wxx]), p.integer(-1)]);
    let prob = PdeProblem::new(&[eq], &[w], &p)
        .unwrap()
        .on(dom, &p)
        .unwrap()
        .ic(w0, zero, &p)
        .unwrap();
    let s = green_solve(&prob, &p).unwrap();
    show_sol(&s, &p);
    assert_eq!(s.status, VerificationStatus::ExactlyVerified);
}
