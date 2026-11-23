#![cfg(test)]
use crate::kernel::{Domain, ExprPool};

#[test]
fn probe() {
    let pool = ExprPool::new();
    let p = &pool;
    let n = p.symbol("n", Domain::Integer);
    let x = p.symbol("x", Domain::Real);
    let l = p.symbol("L", Domain::Positive);
    let k = p.symbol("k", Domain::Positive);
    let t = p.symbol("t", Domain::Positive);
    let xi = p.symbol("xi", Domain::Real);
    let s = p.symbol("s", Domain::Positive);
    let mu = p.symbol("mu", Domain::Positive);
    let pi = p.symbol("pi", Domain::Real);
    let simp = |e| crate::simplify::engine::simplify(e, p).value;
    let sh = |e| p.display(e).to_string();
    println!("sin(n pi) = {}", sh(simp(p.func("sin", vec![p.mul(vec![n, pi])]))));
    println!("cos(n pi) = {}", sh(simp(p.func("cos", vec![p.mul(vec![n, pi])]))));
    let half = p.add(vec![n, p.rational(-1, 2)]);
    println!("cos((n-1/2) pi) = {}", sh(simp(p.func("cos", vec![p.mul(vec![half, pi])]))));
    println!("sin((n-1/2) pi) = {}", sh(simp(p.func("sin", vec![p.mul(vec![half, pi])]))));
    let zt = crate::simplify::zero::zero_test(p.func("sin", vec![p.mul(vec![n, pi])]), p, None);
    println!("zt sin(n pi) {:?}", zt.status);
    let g = p.func("exp", vec![p.mul(vec![p.integer(-4), p.pow(pi, p.integer(2)), k, p.pow(xi, p.integer(2)), t])]);
    match crate::transform::inverse_fourier_transform(g, xi, x, p) {
        Ok(r) => println!("IFT gauss = {}", sh(r)),
        Err(e) => println!("IFT err {e}"),
    }
    let a2 = p.func("exp", vec![p.mul(vec![p.integer(-2), pi, p.func("abs", vec![xi]), t])]);
    match crate::transform::inverse_fourier_transform(a2, xi, x, p) {
        Ok(r) => println!("IFT poisson = {}", sh(r)),
        Err(e) => println!("IFT err {e}"),
    }
    let f0 = p.func("exp", vec![p.mul(vec![p.integer(-1), p.pow(x, p.integer(2))])]);
    match crate::transform::fourier_transform(f0, x, xi, p) {
        Ok(r) => println!("FT e^-x^2 = {}", sh(r)),
        Err(e) => println!("FT err {e}"),
    }
    let sq = p.func("sqrt", vec![p.mul(vec![s, p.pow(k, p.integer(-1))])]);
    let e1 = p.mul(vec![p.func("exp", vec![p.mul(vec![p.integer(-1), x, sq])]), p.pow(s, p.integer(-1))]);
    match crate::transform::inverse_laplace_transform(e1, s, t, p) {
        Ok(r) => println!("ILT erfc = {}", sh(r)),
        Err(e) => println!("ILT err {e}"),
    }
    let y = p.symbol("y", Domain::Real);
    let (inp, _yp, ypp) = crate::ode::dsolve::OdeInput::second_order(x, y, p);
    let inp = inp.with_equation(p.add(vec![ypp, p.mul(vec![p.pow(mu, p.integer(2)), y])]));
    match crate::ode::dsolve::dsolve(&inp, p) {
        Ok(r) => println!("dsolve = {} [{}]", sh(r.solutions[0].y_of_x), r.solutions[0].method),
        Err(e) => println!("dsolve err {e}"),
    }
    let (inp, yp, ypp) = crate::ode::dsolve::OdeInput::second_order(x, y, p);
    let inp = inp.with_equation(p.add(vec![
        p.mul(vec![p.pow(x, p.integer(2)), ypp]),
        p.mul(vec![x, yp]),
        p.mul(vec![p.pow(mu, p.integer(2)), y]),
    ]));
    match crate::ode::dsolve::dsolve(&inp, p) {
        Ok(r) => println!("dsolve euler = {} [{}]", sh(r.solutions[0].y_of_x), r.solutions[0].method),
        Err(e) => println!("dsolve err {e}"),
    }
    let (inp, _yp, ypp) = crate::ode::dsolve::OdeInput::second_order(x, y, p);
    let inp = inp.with_equation(p.add(vec![ypp, p.mul(vec![p.integer(-1), p.pow(mu, p.integer(2)), y])]));
    match crate::ode::dsolve::dsolve(&inp, p) {
        Ok(r) => println!("dsolve hyp = {} [{}]", sh(r.solutions[0].y_of_x), r.solutions[0].method),
        Err(e) => println!("dsolve err {e}"),
    }
    let s2 = p.pow(p.func("sin", vec![p.mul(vec![mu, x])]), p.integer(2));
    match crate::integrate::integrate_definite(s2, x, p.integer(0), l, p) {
        Ok(r) => {
            println!("int sin^2 = {}", sh(r.value));
            let m = std::collections::HashMap::from([(mu, p.mul(vec![n, pi, p.pow(l, p.integer(-1))]))]);
            println!("  at n pi/L = {}", sh(simp(crate::kernel::subs::subs(r.value, &m, p))));
        }
        Err(e) => println!("int err {e}"),
    }
    let b = p.func("bessel_j", vec![p.integer(2), p.mul(vec![mu, x])]);
    println!("bessel_j(2, mu x) = {}  d/dx = {}", sh(b), sh(crate::diff::diff(b, x, p).unwrap().value));
    let b0 = p.func("bessel_j0", vec![p.mul(vec![mu, x])]);
    let d1 = crate::diff::diff(b0, x, p).unwrap().value;
    let d2 = crate::diff::diff(d1, x, p).unwrap().value;
    let res = p.add(vec![
        p.mul(vec![p.pow(x, p.integer(2)), d2]),
        p.mul(vec![x, d1]),
        p.mul(vec![p.pow(mu, p.integer(2)), p.pow(x, p.integer(2)), b0]),
    ]);
    let zt = crate::simplify::zero::zero_test(res, p, None);
    println!("bessel residual {:?} {}", zt.status, sh(simp(res)));
    let leg = p.func("legendre_p", vec![n, x]);
    let d1 = crate::diff::diff(leg, x, p).unwrap().value;
    println!("legendre d/dx = {}", sh(d1));
    let _ = (l, k, t, s);
}

#[test]
fn probe_bessel_ball() {
    let pool = ExprPool::new();
    let p = &pool;
    let mu = p.symbol("mu", Domain::Positive);
    let j2 = crate::specfun::bessel(crate::specfun::BesselKind::J, p.integer(2), mu, false, p).unwrap();
    let d1 = crate::diff::diff(j2, mu, p).unwrap().value;
    let d2 = crate::diff::diff(d1, mu, p).unwrap().value;
    for e in [j2, d1, d2] {
        let mut ie = crate::ball::IntervalEval::new(128);
        ie.bind(mu, crate::ball::ArbBall::from_f64(0.0, 128));
        let v = ie.eval(e, p);
        let mut ie2 = crate::ball::IntervalEval::new(128);
        ie2.bind(
            mu,
            crate::ball::ArbBall {
                mid: rug::Float::with_val(136, 0.125),
                rad: rug::Float::with_val(128, 0.125),
                prec: 128,
            },
        );
        let w = ie2.eval(e, p);
        println!(
            "{} : at0 {:?} on[0,.25] {:?}",
            p.display(e),
            v.map(|b| (b.lo().to_f64(), b.hi().to_f64(), b.is_exact())),
            w.map(|b| (b.lo().to_f64(), b.hi().to_f64()))
        );
    }
}
