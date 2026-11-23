//! The classical singular problems: Bessel on a disk, and the polynomial
//! families (Legendre, Chebyshev, Hermite, Laguerre).

use super::regular::functional;
use super::*;
use crate::orthopoly::{orthopoly, OrthoFamily};
use crate::specfun::{bessel, BesselKind};

/// `e / template` when it is a constant (free of `x`), simplified.
fn ratio(e: ExprId, template: ExprId, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let c = simp(div(e, template, pool), pool);
    (!mentions_var(c, x, pool) && !proved_zero(c, pool)).then_some(c)
}

fn same_end(e: ExprId, v: ExprId, pool: &ExprPool) -> bool {
    e == v || proved_zero(sub(e, v, pool), pool)
}

/// Recognise a singular problem; `None` when it is not one of the classical
/// ones (the caller then treats it as regular, or refuses a `Bounded` /
/// `Decay` condition).
pub(super) fn recognise(
    problem: &SturmLiouvilleProblem,
    n: ExprId,
    opts: &SlOptions,
    pool: &ExprPool,
) -> Result<Option<Eigensystem>, PdeError> {
    let SlBoundary::Separated { left, right } = problem.boundary else {
        return Ok(None);
    };
    let x = problem.var;
    let p = simp(problem.p, pool);
    let q = simp(problem.q, pool);
    let w = simp(problem.w, pool);
    let (a, b) = (problem.a, problem.b);
    let oo = pool.pos_infinity();
    let zero = pool.integer(0);
    let one = pool.integer(1);
    let minus_one = pool.integer(-1);

    // Bessel: p = c·x, w = c·x, q = −c ν²/x on (0, R), bounded at 0.
    if left == EndCondition::Bounded
        && same_end(a, zero, pool)
        && right.coefficients(pool).is_some()
        && !is_infinite(b, pool)
    {
        if let Some(c) = ratio(p, x, x, pool) {
            if proved_zero(sub(simp(div(w, x, pool), pool), c, pool), pool) {
                let nu2 = simp(neg(div(pool.mul(vec![q, x]), c, pool), pool), pool);
                if !mentions_var(nu2, x, pool) {
                    return bessel_problem(problem, c, nu2, right, n, opts, pool).map(Some);
                }
            }
        }
    }
    let bounded_both = left == EndCondition::Bounded && right == EndCondition::Bounded;
    if bounded_both && same_end(a, minus_one, pool) && same_end(b, one, pool) && proved_zero(q, pool) {
        let one_minus = sub(one, pool.pow(x, pool.integer(2)), pool);
        // Legendre: p = c(1 − x²), w = c.
        if let Some(c) = ratio(p, one_minus, x, pool) {
            if proved_zero(sub(w, c, pool), pool) {
                return polynomial_problem(problem, OrthoFamily::Legendre, c, n, opts, pool).map(Some);
            }
        }
        // Chebyshev: p = c√(1 − x²), w = c/√(1 − x²).
        let s = pool.func("sqrt", vec![one_minus]);
        if let Some(c) = ratio(p, s, x, pool) {
            if proved_zero(sub(simp(pool.mul(vec![w, s]), pool), c, pool), pool) {
                return polynomial_problem(problem, OrthoFamily::ChebyshevT, c, n, opts, pool)
                    .map(Some);
            }
        }
    }
    let gauss = pool.func("exp", vec![neg(pool.pow(x, pool.integer(2)), pool)]);
    if left == EndCondition::Decay
        && right == EndCondition::Decay
        && same_end(a, neg(oo, pool), pool)
        && b == oo
        && proved_zero(q, pool)
    {
        if let Some(c) = ratio(p, gauss, x, pool) {
            if proved_zero(sub(simp(div(w, gauss, pool), pool), c, pool), pool) {
                return polynomial_problem(problem, OrthoFamily::HermiteH, c, n, opts, pool).map(Some);
            }
        }
    }
    let e = pool.func("exp", vec![neg(x, pool)]);
    if left == EndCondition::Bounded
        && right == EndCondition::Decay
        && same_end(a, zero, pool)
        && b == oo
        && proved_zero(q, pool)
    {
        if let Some(c) = ratio(p, pool.mul(vec![x, e]), x, pool) {
            if proved_zero(sub(simp(div(w, e, pool), pool), c, pool), pool) {
                return polynomial_problem(problem, OrthoFamily::Laguerre, c, n, opts, pool).map(Some);
            }
        }
    }
    Ok(None)
}

/// The order `ν ≥ 0` from `ν²`.
fn order_from_square(nu2: ExprId, side: &mut Vec<ExprId>, pool: &ExprPool) -> Result<ExprId, PdeError> {
    if let ExprData::Pow { base, exp } = pool.get(nu2) {
        if exp == pool.integer(2) {
            match super::sign(base, pool) {
                Sgn::Pos | Sgn::Zero => return Ok(base),
                Sgn::Neg => return Ok(simp(neg(base, pool), pool)),
                Sgn::Unknown => {
                    side.push(pool.pred_ge(base, pool.integer(0)));
                    return Ok(base);
                }
            }
        }
    }
    match super::sign(nu2, pool) {
        Sgn::Neg => Err(PdeError::SlUnsupported(format!(
            "Bessel order squared {} is negative",
            pool.display(nu2)
        ))),
        Sgn::Zero => Ok(pool.integer(0)),
        Sgn::Pos => Ok(simp(pool.func("sqrt", vec![nu2]), pool)),
        Sgn::Unknown => {
            side.push(pool.pred_ge(nu2, pool.integer(0)));
            Ok(simp(pool.func("sqrt", vec![nu2]), pool))
        }
    }
}

fn bessel_problem(
    problem: &SturmLiouvilleProblem,
    c: ExprId,
    nu2: ExprId,
    right: EndCondition,
    n: ExprId,
    opts: &SlOptions,
    pool: &ExprPool,
) -> Result<Eigensystem, PdeError> {
    let x = problem.var;
    let radius = problem.b;
    let rc = right.coefficients(pool).expect("regular outer end");
    let mut side = Vec::new();
    let mut notes = Vec::new();
    match super::sign(c, pool) {
        Sgn::Pos => {}
        Sgn::Neg | Sgn::Zero => {
            return Err(PdeError::SlUnsupported(
                "Bessel problem with a non-positive p/x".into(),
            ))
        }
        Sgn::Unknown => side.push(pool.pred_gt(c, pool.integer(0))),
    }
    if super::sign(radius, pool) != Sgn::Pos {
        side.push(pool.pred_gt(radius, pool.integer(0)));
    }
    let order = order_from_square(nu2, &mut side, pool)?;
    let p = simp(problem.p, pool);
    let q = simp(problem.q, pool);
    let w = simp(problem.w, pool);
    let avoid = [p, q, w, problem.a, radius, x, n];
    let mu = fresh_symbol("mu", Domain::Positive, &avoid, pool);
    let kappa = fresh_symbol("kappa", Domain::Positive, &avoid, pool);
    let bes = |kind: BesselKind, arg: ExprId| -> Result<ExprId, PdeError> {
        bessel(kind, order, arg, false, pool)
            .map_err(|e| PdeError::SlUnsupported(format!("Bessel function: {e}")))
    };
    let xf = bes(BesselKind::J, pool.mul(vec![mu, x]))?;
    let lam = simp(pool.pow(mu, pool.integer(2)), pool);
    let dfun = functional(xf, x, radius, rc, pool)?;
    if proved_zero(dfun, pool) {
        return Err(PdeError::SlDegenerate("the Bessel boundary condition vanishes identically".into()));
    }
    let ode = zstatus(ode_residual(x, p, q, w, lam, xf, pool)?, pool);
    if ode == ZeroStatus::ProvedNonzero {
        return Err(PdeError::SlVerificationFailed(format!(
            "Bessel eigenfunction {}",
            pool.display(xf)
        )));
    }
    let frequency = root_of_family(dfun, mu, n, 0, pool)?;
    // ∫₀^R r J_ν(μr)² dr = (R²/2)[J_ν'(μR)² + (1 − ν²/(μR)²) J_ν(μR)²].
    let z = fresh_symbol("__sl_z", Domain::Positive, &avoid, pool);
    let jz = bes(BesselKind::J, z)?;
    let djz = d(jz, z, pool)?;
    let mur = pool.mul(vec![mu, radius]);
    let norm_mu = simp(
        pool.mul(vec![
            c,
            pool.pow(radius, pool.integer(2)),
            pool.rational(1, 2),
            pool.add(vec![
                pool.pow(subs1(djz, z, mur, pool), pool.integer(2)),
                pool.mul(vec![
                    sub(
                        pool.integer(1),
                        div(pool.pow(order, pool.integer(2)), pool.pow(mur, pool.integer(2)), pool),
                        pool,
                    ),
                    pool.pow(subs1(jz, z, mur, pool), pool.integer(2)),
                ]),
            ]),
        ]),
        pool,
    );
    let fam = EigenFamily {
        index: n,
        first: 1,
        frequency_var: mu,
        frequency,
        eigenvalue: simp(subs1(lam, mu, frequency, pool), pool),
        eigenvalue_of_frequency: lam,
        eigenfunctions: vec![simp(subs1(xf, mu, frequency, pool), pool)],
        eigenfunctions_of_frequency: vec![xf],
        norms_squared: vec![Some(simp(subs1(norm_mu, mu, frequency, pool), pool))],
        closed_form: false,
        defining_equation: Some(dfun),
        enclosures: Vec::new(),
        verification: EigenVerification {
            status: combine(&[ode]),
            ode,
            left: ZeroStatus::ProvedZero,
            right: ZeroStatus::ProvedZero,
            method: "defining_equation",
            members: Vec::new(),
        },
    };
    notes.push(
        "R = J_ν(μr) is the bounded solution at r = 0 (Y_ν is unbounded there); the outer \
         condition holds at the roots of the defining equation by construction"
            .into(),
    );
    // λ = 0: the bounded solution is r^ν.
    let mut modes = Vec::new();
    let x0 = simp(pool.pow(x, order), pool);
    let d0 = functional(x0, x, radius, rc, pool)?;
    let cond = if proved_zero(d0, pool) {
        Some(None)
    } else {
        match super::sign(d0, pool) {
            Sgn::Pos | Sgn::Neg => None,
            _ => Some(Some(pool.pred_eq(d0, pool.integer(0)))),
        }
    };
    if let Some(condition) = cond {
        let st = zstatus(ode_residual(x, p, q, w, pool.integer(0), x0, pool)?, pool);
        modes.push(EigenMode {
            eigenvalue: pool.integer(0),
            eigenfunction: x0,
            norm_squared: crate::integrate::integrate_definite(
                pool.mul(vec![w, pool.pow(x0, pool.integer(2))]),
                x,
                pool.integer(0),
                radius,
                pool,
            )
            .ok()
            .map(|r| simp(r.value, pool)),
            condition,
            verification: status_of(st),
        });
    }
    // λ = −κ²: the bounded solution is I_ν(κr).
    let (aa, bb) = rc;
    let sb = super::sign(pool.mul(vec![aa, bb]), pool);
    let lower = if matches!(sb, Sgn::Pos | Sgn::Zero) {
        LowerSpectrum::None
    } else {
        let zf = bes(BesselKind::I, pool.mul(vec![kappa, x]))?;
        let dl = functional(zf, x, radius, rc, pool)?;
        let lam_k = simp(neg(pool.pow(kappa, pool.integer(2)), pool), pool);
        let fr = root_of_family(dl, kappa, n, 0, pool)?;
        let st = zstatus(ode_residual(x, p, q, w, lam_k, zf, pool)?, pool);
        let lf = EigenFamily {
            index: n,
            first: 1,
            frequency_var: kappa,
            frequency: fr,
            eigenvalue: simp(subs1(lam_k, kappa, fr, pool), pool),
            eigenvalue_of_frequency: lam_k,
            eigenfunctions: vec![simp(subs1(zf, kappa, fr, pool), pool)],
            eigenfunctions_of_frequency: vec![zf],
            norms_squared: vec![None],
            closed_form: false,
            defining_equation: Some(dl),
            enclosures: Vec::new(),
            verification: EigenVerification {
                status: combine(&[st]),
                ode: st,
                left: ZeroStatus::ProvedZero,
                right: ZeroStatus::ProvedZero,
                method: "defining_equation",
                members: Vec::new(),
            },
        };
        if sb == Sgn::Neg {
            match lf.enclosures(1, &HashMap::new(), 64, pool) {
                Ok(enc) => {
                    let mut lf = lf;
                    lf.enclosures = enc;
                    LowerSpectrum::Present {
                        family: Box::new(lf),
                        count: None,
                    }
                }
                Err(_) => LowerSpectrum::Conditional {
                    condition: pool.pred_ge(simp(pool.mul(vec![aa, bb]), pool), pool.integer(0)),
                    family: Some(Box::new(lf)),
                },
            }
        } else {
            LowerSpectrum::Conditional {
                condition: pool.pred_ge(simp(pool.mul(vec![aa, bb]), pool), pool.integer(0)),
                family: Some(Box::new(lf)),
            }
        }
    };
    let mut sys = Eigensystem {
        kind: "bessel",
        var: x,
        weight: w,
        interval: (problem.a, radius),
        threshold: pool.integer(0),
        families: vec![fam],
        modes,
        lower,
        side_conditions: side,
        notes,
    };
    fill_enclosures(&mut sys, opts, pool);
    Ok(sys)
}

/// `(λ(n), ∫ w X_n²)` for a polynomial family, `w = c·w₀`.
fn poly_data(family: OrthoFamily, nn: ExprId, c: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let pi = pool.symbol("pi", Domain::Real);
    match family {
        OrthoFamily::Legendre => (
            pool.mul(vec![nn, pool.add(vec![nn, pool.integer(1)])]),
            pool.mul(vec![
                pool.integer(2),
                c,
                pool.pow(pool.add(vec![pool.mul(vec![pool.integer(2), nn]), pool.integer(1)]), pool.integer(-1)),
            ]),
        ),
        OrthoFamily::ChebyshevT => (
            pool.pow(nn, pool.integer(2)),
            pool.mul(vec![c, pi, pool.rational(1, 2)]),
        ),
        OrthoFamily::HermiteH => (
            pool.mul(vec![pool.integer(2), nn]),
            pool.mul(vec![
                c,
                pool.func("sqrt", vec![pi]),
                pool.pow(pool.integer(2), nn),
                pool.func("factorial", vec![nn]),
            ]),
        ),
        _ => (nn, c),
    }
}

fn polynomial_problem(
    problem: &SturmLiouvilleProblem,
    family: OrthoFamily,
    c: ExprId,
    n: ExprId,
    opts: &SlOptions,
    pool: &ExprPool,
) -> Result<Eigensystem, PdeError> {
    let x = problem.var;
    let p = simp(problem.p, pool);
    let q = simp(problem.q, pool);
    let w = simp(problem.w, pool);
    let mut side = Vec::new();
    match super::sign(c, pool) {
        Sgn::Pos => {}
        Sgn::Neg | Sgn::Zero => {
            return Err(PdeError::SlUnsupported(
                "a classical polynomial problem with a non-positive weight".into(),
            ))
        }
        Sgn::Unknown => side.push(pool.pred_gt(c, pool.integer(0))),
    }
    let avoid = [p, q, w, problem.a, problem.b, x, n];
    let nu = fresh_symbol("nu", Domain::NonNegative, &avoid, pool);
    let (lam_nu, _) = poly_data(family, nu, c, pool);
    let (lam_n, norm_n) = poly_data(family, n, c, pool);
    let first = if family == OrthoFamily::ChebyshevT { 1 } else { 0 };
    let node = orthopoly(family, n, &[], x, false, pool)
        .map_err(|e| PdeError::SlUnsupported(format!("orthogonal polynomial: {e}")))?;
    // Exact checks on the first members with the explicit polynomials.
    let mut members = Vec::new();
    let mut statuses = Vec::new();
    for k in first..first + opts.check_members.max(1) as i64 {
        let kk = pool.integer(k);
        let xk = orthopoly(family, kk, &[], x, true, pool)
            .map_err(|e| PdeError::SlUnsupported(format!("orthogonal polynomial: {e}")))?;
        let lam_k = simp(subs1(lam_n, n, kk, pool), pool);
        let st = zstatus(ode_residual(x, p, q, w, lam_k, xk, pool)?, pool);
        if st == ZeroStatus::ProvedNonzero {
            return Err(PdeError::SlVerificationFailed(format!(
                "{}({k}, x) does not satisfy the equation",
                family.name()
            )));
        }
        statuses.push(st);
        members.push(k);
    }
    // And for symbolic n, when the derivative identities close.
    let symbolic = zstatus(ode_residual(x, p, q, w, lam_n, node, pool)?, pool);
    let (status, method) = if symbolic == ZeroStatus::ProvedZero {
        (VerificationStatus::ExactlyVerified, "closed_form")
    } else {
        (combine(&statuses), "members")
    };
    let fam = EigenFamily {
        index: n,
        first,
        frequency_var: nu,
        frequency: n,
        eigenvalue: simp(lam_n, pool),
        eigenvalue_of_frequency: simp(lam_nu, pool),
        eigenfunctions: vec![node],
        eigenfunctions_of_frequency: vec![subs1(node, n, nu, pool)],
        norms_squared: vec![Some(simp(norm_n, pool))],
        closed_form: true,
        defining_equation: None,
        enclosures: Vec::new(),
        verification: EigenVerification {
            status,
            ode: if symbolic == ZeroStatus::ProvedZero {
                symbolic
            } else {
                combine_zero(&statuses)
            },
            left: ZeroStatus::ProvedZero,
            right: ZeroStatus::ProvedZero,
            method,
            members: if method == "members" { members } else { Vec::new() },
        },
    };
    let mut modes = Vec::new();
    if family == OrthoFamily::ChebyshevT {
        let pi = pool.symbol("pi", Domain::Real);
        modes.push(EigenMode {
            eigenvalue: pool.integer(0),
            eigenfunction: pool.integer(1),
            norm_squared: Some(simp(pool.mul(vec![c, pi]), pool)),
            condition: None,
            verification: status_of(zstatus(
                ode_residual(x, p, q, w, pool.integer(0), pool.integer(1), pool)?,
                pool,
            )),
        });
    }
    let kind = match family {
        OrthoFamily::Legendre => "legendre",
        OrthoFamily::ChebyshevT => "chebyshev",
        OrthoFamily::HermiteH => "hermite",
        _ => "laguerre",
    };
    let mut sys = Eigensystem {
        kind,
        var: x,
        weight: w,
        interval: (problem.a, problem.b),
        threshold: pool.integer(0),
        families: vec![fam],
        modes,
        lower: LowerSpectrum::None,
        side_conditions: side,
        notes: vec![format!(
            "the {kind} polynomials are the only eigenfunctions satisfying the end conditions \
             (bounded / of polynomial growth); the conditions hold by polynomiality"
        )],
    };
    fill_enclosures(&mut sys, opts, pool);
    Ok(sys)
}

fn combine_zero(s: &[ZeroStatus]) -> ZeroStatus {
    if s.iter().all(|z| *z == ZeroStatus::ProvedZero) {
        ZeroStatus::ProvedZero
    } else if s.contains(&ZeroStatus::ProvedNonzero) {
        ZeroStatus::ProvedNonzero
    } else {
        ZeroStatus::Unknown
    }
}
