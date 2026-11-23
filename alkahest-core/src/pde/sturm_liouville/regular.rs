//! Regular problems: constant coefficients, Euler–Cauchy, and whatever else
//! `dsolve` solves — all through one 2×2 boundary determinant on a basis.

use super::*;

/// A fundamental system parametrised by a frequency.
pub(super) struct Basis {
    /// The positive frequency symbol (`ν`, `ω`, `μ`).
    pub freq: ExprId,
    /// `y₁`, `y₂` above the threshold (oscillatory).
    pub y: [ExprId; 2],
    /// `λ` as a function of `freq`.
    pub lam: ExprId,
    /// `y₁`, `y₂` at `freq = 0` (the threshold).
    pub y0: [ExprId; 2],
    /// `λ₀`.
    pub lam0: ExprId,
    /// The basis below the threshold, in its own symbol `κ`, with `λ(κ)`:
    /// `(κ, [z₁, z₂], λ(κ))`.
    pub lower: Option<(ExprId, [ExprId; 2], ExprId)>,
    /// `"regular_constant"`, `"regular"`.
    pub kind: &'static str,
}

/// `α f(end) + β f'(end)`.
pub(super) fn functional(
    f: ExprId,
    x: ExprId,
    end: ExprId,
    (alpha, beta): (ExprId, ExprId),
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let df = d(f, x, pool)?;
    Ok(simp(
        pool.add(vec![
            pool.mul(vec![alpha, subs1(f, x, end, pool)]),
            pool.mul(vec![beta, subs1(df, x, end, pool)]),
        ]),
        pool,
    ))
}

/// The structure of a determinant `D(ν) = c_s·T₁(A) + c_c·T₂(A)` where
/// `T₁, T₂` are `sin, cos` (or `sinh, cosh`) of one argument `A = κν`.
pub(super) struct TrigShape {
    /// `A / ν`.
    pub kappa: ExprId,
    /// Coefficient of `sin A` (or `sinh A`).
    pub cs: ExprId,
    /// Coefficient of `cos A` (or `cosh A`).
    pub cc: ExprId,
}

/// Decompose `dfun` as `cs·s(A) + cc·c(A)` (`(s, c) = names`), with `cs`,
/// `cc` free of trigonometric functions of `freq`.
pub(super) fn trig_shape(
    dfun: ExprId,
    freq: ExprId,
    names: (&str, &str),
    pool: &ExprPool,
) -> Option<TrigShape> {
    // Every s/c node whose argument mentions the frequency.
    let mut args: Vec<ExprId> = Vec::new();
    let mut stack = vec![dfun];
    let mut seen = std::collections::HashSet::new();
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        if let ExprData::Func { name, args: a } = pool.get(e) {
            if (name == names.0 || name == names.1)
                && a.len() == 1
                && mentions_var(a[0], freq, pool)
            {
                if !args.contains(&a[0]) {
                    args.push(a[0]);
                }
                continue;
            }
        }
        super::super::jet::push_children(e, pool, &mut stack);
    }
    if args.len() != 1 {
        return None;
    }
    let arg = args[0];
    let kappa = simp(div(arg, freq, pool), pool);
    if mentions_var(kappa, freq, pool) {
        return None;
    }
    let s_sym = fresh_symbol("__sl_S", Domain::Real, &[dfun], pool);
    let c_sym = fresh_symbol("__sl_C", Domain::Real, &[dfun], pool);
    let e = replace_node(dfun, pool.func(names.0, vec![arg]), s_sym, pool);
    let e = replace_node(e, pool.func(names.1, vec![arg]), c_sym, pool);
    if mentions_var(e, freq, pool) {
        // still fine: polynomial coefficients in freq
    }
    let cs = simp(d(e, s_sym, pool).ok()?, pool);
    let cc = simp(d(e, c_sym, pool).ok()?, pool);
    for c in [cs, cc] {
        if mentions_var(c, s_sym, pool) || mentions_var(c, c_sym, pool) {
            return None;
        }
        // No other function of the frequency may hide in a coefficient.
        if has_func_of(c, freq, pool) {
            return None;
        }
    }
    let rest = pool.add(vec![
        e,
        neg(pool.mul(vec![cs, s_sym]), pool),
        neg(pool.mul(vec![cc, c_sym]), pool),
    ]);
    if !proved_zero(rest, pool) {
        return None;
    }
    Some(TrigShape { kappa, cs, cc })
}

fn has_func_of(e: ExprId, v: ExprId, pool: &ExprPool) -> bool {
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let ExprData::Func { .. } = pool.get(n) {
            if mentions_var(n, v, pool) {
                return true;
            }
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
    false
}

/// Rewrite `sin(A + rπ)`, `cos(A + rπ)` (`r` rational) by the addition
/// formulas so that `sin(nπ) = 0`, `cos(nπ) = (−1)ⁿ` can fold; used only to
/// *prove* closed-form boundary residuals zero.
pub(super) fn reduce_pi_shifts(e: ExprId, pool: &ExprPool) -> ExprId {
    let pi = pool.symbol("pi", Domain::Real);
    fn go(e: ExprId, pi: ExprId, pool: &ExprPool) -> ExprId {
        match pool.get(e) {
            ExprData::Add(xs) => pool.add(xs.iter().map(|&x| go(x, pi, pool)).collect()),
            ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| go(x, pi, pool)).collect()),
            ExprData::Pow { base, exp } => pool.pow(go(base, pi, pool), go(exp, pi, pool)),
            ExprData::Func { name, args }
                if (name == "sin" || name == "cos") && args.len() == 1 =>
            {
                let arg = crate::simplify::engine::simplify_expanded(args[0], pool).value;
                let terms = match pool.get(arg) {
                    ExprData::Add(xs) => xs,
                    _ => vec![arg],
                };
                let mut shift = Vec::new();
                let mut rest = Vec::new();
                for t in terms {
                    let r = simp(div(t, pi, pool), pool);
                    let is_rat = pool.with(r, |d| {
                        matches!(d, ExprData::Integer(_) | ExprData::Rational(_))
                    });
                    if is_rat {
                        shift.push(t);
                    } else {
                        rest.push(t);
                    }
                }
                if shift.is_empty() || rest.is_empty() {
                    return pool.func(name, vec![arg]);
                }
                let a = pool.add(rest);
                let b = simp(pool.add(shift), pool);
                let (sa, ca) = (pool.func("sin", vec![a]), pool.func("cos", vec![a]));
                let (sb, cb) = (
                    simp(pool.func("sin", vec![b]), pool),
                    simp(pool.func("cos", vec![b]), pool),
                );
                if name == "sin" {
                    pool.add(vec![pool.mul(vec![sa, cb]), pool.mul(vec![ca, sb])])
                } else {
                    pool.add(vec![
                        pool.mul(vec![ca, cb]),
                        neg(pool.mul(vec![sa, sb]), pool),
                    ])
                }
            }
            ExprData::Func { name, args } => {
                pool.func(name, args.iter().map(|&x| go(x, pi, pool)).collect())
            }
            _ => e,
        }
    }
    simp(go(e, pi, pool), pool)
}

/// Exact status of a closed-form residual (after π-shift reduction).
fn closed_status(e: ExprId, pool: &ExprPool) -> ZeroStatus {
    let s = zstatus(e, pool);
    if s == ZeroStatus::ProvedZero {
        return s;
    }
    let r = reduce_pi_shifts(simp(e, pool), pool);
    let r2 = zstatus(r, pool);
    if r2 == ZeroStatus::ProvedZero {
        return r2;
    }
    // A residual in an integer index is never refuted by sampling at
    // non-integer points: only a proof counts here.
    match s {
        ZeroStatus::ProvedNonzero => ZeroStatus::Unknown,
        other => other,
    }
}

/// The eigen-family on a basis, for separated regular conditions.
#[allow(clippy::too_many_arguments)]
pub(super) fn separated_family(
    problem: &SturmLiouvilleProblem,
    p: ExprId,
    q: ExprId,
    w: ExprId,
    freq: ExprId,
    y: [ExprId; 2],
    lam_of_freq: ExprId,
    left: (ExprId, ExprId),
    right: (ExprId, ExprId),
    n: ExprId,
    names: (&str, &str),
    side: &mut Vec<ExprId>,
    notes: &mut Vec<String>,
    pool: &ExprPool,
) -> Result<Option<EigenFamily>, PdeError> {
    let x = problem.var;
    let (a, b) = (problem.a, problem.b);
    let ba = [
        functional(y[0], x, a, left, pool)?,
        functional(y[1], x, a, left, pool)?,
    ];
    let bb = [
        functional(y[0], x, b, right, pool)?,
        functional(y[1], x, b, right, pool)?,
    ];
    let dfun = simp(
        sub(
            pool.mul(vec![ba[0], bb[1]]),
            pool.mul(vec![ba[1], bb[0]]),
            pool,
        ),
        pool,
    );
    if proved_zero(dfun, pool) {
        return Err(PdeError::SlDegenerate(format!(
            "the boundary determinant vanishes identically in {}",
            pool.display(freq)
        )));
    }
    // X = B_a(y₂)·y₁ − B_a(y₁)·y₂ satisfies the left condition identically;
    // drop a factor when one coefficient vanishes.
    let (xf, scale) = if proved_zero(ba[0], pool) {
        (y[0], ba[1])
    } else if proved_zero(ba[1], pool) {
        (y[1], neg(ba[0], pool))
    } else {
        (
            simp(
                sub(
                    pool.mul(vec![ba[1], y[0]]),
                    pool.mul(vec![ba[0], y[1]]),
                    pool,
                ),
                pool,
            ),
            pool.integer(1),
        )
    };
    let shape = trig_shape(dfun, freq, names, pool);
    let hyperbolic = names.0 == "sinh";
    if hyperbolic {
        // Below the threshold only the *absence* of roots is decided here.
        if let Some(sh) = &shape {
            let only = |c: ExprId, other: ExprId| {
                proved_zero(other, pool) && matches!(super::sign(c, pool), Sgn::Pos | Sgn::Neg)
            };
            if only(sh.cs, sh.cc) || only(sh.cc, sh.cs) {
                return Ok(None);
            }
        }
    }
    let pi = pool.symbol("pi", Domain::Real);
    let mut closed: Option<ExprId> = None;
    if let (false, Some(sh)) = (hyperbolic, &shape) {
        let kappa = match super::sign(sh.kappa, pool) {
            Sgn::Pos => Some(sh.kappa),
            Sgn::Neg => Some(simp(neg(sh.kappa, pool), pool)),
            _ => {
                side.push(pool.pred_gt(sh.kappa, pool.integer(0)));
                Some(sh.kappa)
            }
        };
        let kappa = kappa.unwrap_or(sh.kappa);
        let (cs0, cc0) = (proved_zero(sh.cs, pool), proved_zero(sh.cc, pool));
        let candidate = if cc0 && !cs0 {
            Some((
                simp(
                    pool.mul(vec![n, pi, pool.pow(kappa, pool.integer(-1))]),
                    pool,
                ),
                sh.cs,
            ))
        } else if cs0 && !cc0 {
            Some((
                simp(
                    pool.mul(vec![
                        sub(pool.mul(vec![pool.integer(2), n]), pool.integer(1), pool),
                        pi,
                        pool.pow(pool.mul(vec![pool.integer(2), kappa]), pool.integer(-1)),
                    ]),
                    pool,
                ),
                sh.cc,
            ))
        } else {
            None
        };
        if let Some((fn_, coeff)) = candidate {
            // The coefficient must not vanish at a root (else more roots).
            if !matches!(super::sign(coeff, pool), Sgn::Pos | Sgn::Neg) {
                let at = simp(subs1(coeff, freq, fn_, pool), pool);
                side.push(pool.pred_ne(at, pool.integer(0)));
            }
            closed = Some(fn_);
        } else if !cs0 && !cc0 {
            notes.push(format!(
                "the eigenvalue condition {} = 0 has no closed-form roots; where {} = 0 the roots are \
                 those of {}(...) alone",
                pool.display(dfun),
                pool.display(sh.cs),
                names.1
            ));
        }
    }
    let first = 1;
    let (frequency, closed_form) = match closed {
        Some(c) => (c, true),
        None => (root_of_family(dfun, freq, n, 0, pool)?, false),
    };
    // Verification, identically in the frequency where possible.
    let ode = zstatus(ode_residual(x, p, q, w, lam_of_freq, xf, pool)?, pool);
    let left_st = zstatus(functional(xf, x, a, left, pool)?, pool);
    let right_res = functional(xf, x, b, right, pool)?;
    let (right_st, method) = if closed_form {
        (
            closed_status(subs1(right_res, freq, frequency, pool), pool),
            "closed_form",
        )
    } else {
        // scale·B_b(X) = −D(ν): the residual is the defining equation.
        (
            zstatus(pool.add(vec![pool.mul(vec![scale, right_res]), dfun]), pool),
            "defining_equation",
        )
    };
    let status = combine(&[ode, left_st, right_st]);
    if status == VerificationStatus::Refuted {
        return Err(PdeError::SlVerificationFailed(format!(
            "eigenfunction {} (ode {:?}, left {:?}, right {:?})",
            pool.display(xf),
            ode,
            left_st,
            right_st
        )));
    }
    let eigenfunction = simp(subs1(xf, freq, frequency, pool), pool);
    let eigenvalue = simp(subs1(lam_of_freq, freq, frequency, pool), pool);
    let norm = crate::integrate::integrate_definite(
        pool.mul(vec![w, pool.pow(xf, pool.integer(2))]),
        x,
        a,
        b,
        pool,
    )
    .ok()
    .map(|r| {
        let v = simp(subs1(r.value, freq, frequency, pool), pool);
        if closed_form {
            reduce_pi_shifts(v, pool)
        } else {
            v
        }
    });
    Ok(Some(EigenFamily {
        index: n,
        first,
        frequency_var: freq,
        frequency,
        eigenvalue,
        eigenvalue_of_frequency: lam_of_freq,
        eigenfunctions: vec![eigenfunction],
        eigenfunctions_of_frequency: vec![xf],
        norms_squared: vec![norm],
        closed_form,
        defining_equation: Some(dfun),
        enclosures: Vec::new(),
        verification: EigenVerification {
            status,
            ode,
            left: left_st,
            right: right_st,
            method,
            members: Vec::new(),
        },
    }))
}

/// Positivity of `p` and `w` on the interval: proved, or a side condition.
fn require_positive(
    e: ExprId,
    what: &str,
    problem: &SturmLiouvilleProblem,
    side: &mut Vec<ExprId>,
    pool: &ExprPool,
) -> Result<(), PdeError> {
    let x = problem.var;
    if !mentions_var(e, x, pool) {
        return match super::sign(e, pool) {
            Sgn::Pos => Ok(()),
            Sgn::Neg | Sgn::Zero => Err(PdeError::SlUnsupported(format!(
                "{what} = {} is not positive",
                pool.display(e)
            ))),
            Sgn::Unknown => {
                side.push(pool.pred_gt(e, pool.integer(0)));
                Ok(())
            }
        };
    }
    // Variable: a ball enclosure over the interval when it is numeric.
    let lo = IntervalEval::new(96).eval(problem.a, pool);
    let hi = IntervalEval::new(96).eval(problem.b, pool);
    if let (Some(lo), Some(hi)) = (lo, hi) {
        let (l, h) = (lo.lo(), hi.hi());
        let mid = Float::with_val(128, &l + &h) / 2u32;
        let rad = Float::with_val_round(128, &h - &mid, rug::float::Round::Up).0;
        let mut ie = IntervalEval::new(96);
        ie.bind(x, ArbBall { mid, rad, prec: 96 });
        if let Some(v) = ie.eval(e, pool) {
            if v.lo().is_finite() && v.lo() > 0 {
                return Ok(());
            }
        }
    }
    side.push(pool.pred_gt(e, pool.integer(0)));
    Ok(())
}

/// The isolated eigenpair at the threshold `freq = 0`.
#[allow(clippy::too_many_arguments)]
fn threshold_mode(
    problem: &SturmLiouvilleProblem,
    p: ExprId,
    q: ExprId,
    w: ExprId,
    y0: [ExprId; 2],
    lam0: ExprId,
    left: (ExprId, ExprId),
    right: (ExprId, ExprId),
    pool: &ExprPool,
) -> Result<Option<EigenMode>, PdeError> {
    let x = problem.var;
    let ba = [
        functional(y0[0], x, problem.a, left, pool)?,
        functional(y0[1], x, problem.a, left, pool)?,
    ];
    let bb = [
        functional(y0[0], x, problem.b, right, pool)?,
        functional(y0[1], x, problem.b, right, pool)?,
    ];
    let d0 = simp(
        sub(
            pool.mul(vec![ba[0], bb[1]]),
            pool.mul(vec![ba[1], bb[0]]),
            pool,
        ),
        pool,
    );
    let condition = if proved_zero(d0, pool) {
        None
    } else {
        match super::sign(d0, pool) {
            Sgn::Pos | Sgn::Neg => return Ok(None),
            _ => Some(pool.pred_eq(d0, pool.integer(0))),
        }
    };
    let xf = if proved_zero(ba[0], pool) {
        y0[0]
    } else if proved_zero(ba[1], pool) {
        y0[1]
    } else {
        simp(
            sub(
                pool.mul(vec![ba[1], y0[0]]),
                pool.mul(vec![ba[0], y0[1]]),
                pool,
            ),
            pool,
        )
    };
    let ode = zstatus(ode_residual(x, p, q, w, lam0, xf, pool)?, pool);
    let l = zstatus(functional(xf, x, problem.a, left, pool)?, pool);
    let mut checks = vec![ode, l];
    if condition.is_none() {
        checks.push(zstatus(functional(xf, x, problem.b, right, pool)?, pool));
    }
    let verification = combine(&checks);
    if verification == VerificationStatus::Refuted {
        return Err(PdeError::SlVerificationFailed(format!(
            "threshold eigenfunction {}",
            pool.display(xf)
        )));
    }
    let norm = crate::integrate::integrate_definite(
        pool.mul(vec![w, pool.pow(xf, pool.integer(2))]),
        x,
        problem.a,
        problem.b,
        pool,
    )
    .ok()
    .map(|r| simp(r.value, pool));
    Ok(Some(EigenMode {
        eigenvalue: simp(lam0, pool),
        eigenfunction: xf,
        norm_squared: norm,
        condition,
        verification,
    }))
}

/// Separated regular conditions on a basis: families, threshold mode,
/// lower spectrum.
fn separated(
    problem: &SturmLiouvilleProblem,
    p: ExprId,
    q: ExprId,
    w: ExprId,
    basis: Basis,
    n: ExprId,
    pool: &ExprPool,
) -> Result<Eigensystem, PdeError> {
    let SlBoundary::Separated { left, right } = problem.boundary else {
        unreachable!("separated() on a periodic problem")
    };
    let lc = left.coefficients(pool).expect("regular end");
    let rc = right.coefficients(pool).expect("regular end");
    let mut side = Vec::new();
    let mut notes = Vec::new();
    require_positive(p, "p", problem, &mut side, pool)?;
    require_positive(w, "w", problem, &mut side, pool)?;
    let fam = separated_family(
        problem,
        p,
        q,
        w,
        basis.freq,
        basis.y,
        basis.lam,
        lc,
        rc,
        n,
        ("sin", "cos"),
        &mut side,
        &mut notes,
        pool,
    )?
    .expect("oscillatory family");
    let mut modes = Vec::new();
    if let Some(m) = threshold_mode(problem, p, q, w, basis.y0, basis.lam0, lc, rc, pool)? {
        modes.push(m);
    }
    // Below the threshold.
    let constant = basis.kind == "regular_constant";
    let lam0 = basis.lam0;
    let lower = match basis.lower {
        None => rayleigh(problem, lc, rc, q, lam0, constant, None, pool),
        Some((kappa, z, lam_k)) => {
            let low = separated_family(
                problem,
                p,
                q,
                w,
                kappa,
                z,
                lam_k,
                lc,
                rc,
                n,
                ("sinh", "cosh"),
                &mut Vec::new(),
                &mut Vec::new(),
                pool,
            );
            match low {
                Ok(None) | Err(PdeError::SlDegenerate(_)) => LowerSpectrum::None,
                Ok(Some(f)) => rayleigh(problem, lc, rc, q, lam0, constant, Some(f), pool),
                Err(e) => return Err(e),
            }
        }
    };
    Ok(Eigensystem {
        kind: basis.kind,
        var: problem.var,
        weight: w,
        interval: (problem.a, problem.b),
        threshold: simp(basis.lam0, pool),
        families: vec![fam],
        modes,
        lower,
        side_conditions: side,
        notes,
    })
}

/// The Rayleigh-quotient case analysis for eigenvalues below the threshold.
///
/// `λ ∫wX² = ∫pX'² − ∫qX² + p(b)(α_b/β_b)X(b)² − p(a)(α_a/β_a)X(a)²`, so
/// with `α_a β_a ≤ 0 ≤ α_b β_b` every eigenvalue satisfies `λ ≥ −q/w` for
/// constant coefficients (the threshold) and `λ ≥ 0` when `q ≤ 0` — which
/// is the threshold only when `λ₀ ≤ 0`.
#[allow(clippy::too_many_arguments)]
fn rayleigh(
    problem: &SturmLiouvilleProblem,
    (aa, ba): (ExprId, ExprId),
    (ab, bb): (ExprId, ExprId),
    q: ExprId,
    lam0: ExprId,
    constant: bool,
    family: Option<EigenFamily>,
    pool: &ExprPool,
) -> LowerSpectrum {
    let sa = super::sign(pool.mul(vec![aa, ba]), pool);
    let sb = super::sign(pool.mul(vec![ab, bb]), pool);
    let left_ok = matches!(sa, Sgn::Neg | Sgn::Zero);
    let right_ok = matches!(sb, Sgn::Pos | Sgn::Zero);
    let q_ok = constant
        || (!mentions_var(q, problem.var, pool)
            && matches!(super::sign(q, pool), Sgn::Neg | Sgn::Zero));
    let lam0_ok = constant || matches!(super::sign(lam0, pool), Sgn::Neg | Sgn::Zero);
    if left_ok && right_ok && q_ok && lam0_ok {
        return LowerSpectrum::None;
    }
    // The hypothesis failing does not by itself produce an eigenvalue below
    // the threshold: report one only when a member is certified.
    if let (true, Some(f)) = (matches!(sa, Sgn::Pos) || matches!(sb, Sgn::Neg), &family) {
        if let Ok(enc) = f.enclosures(1, &HashMap::new(), 64, pool) {
            let mut f = f.clone();
            f.enclosures = enc;
            return LowerSpectrum::Present {
                family: Box::new(f),
                count: None,
            };
        }
    }
    let mut conds = Vec::new();
    if !left_ok {
        conds.push(pool.pred_le(simp(pool.mul(vec![aa, ba]), pool), pool.integer(0)));
    }
    if !right_ok {
        conds.push(pool.pred_ge(simp(pool.mul(vec![ab, bb]), pool), pool.integer(0)));
    }
    if !q_ok {
        conds.push(pool.pred_le(q, pool.integer(0)));
    }
    if !lam0_ok {
        conds.push(pool.pred_le(lam0, pool.integer(0)));
    }
    let condition = if conds.len() == 1 {
        conds[0]
    } else {
        pool.pred_and(conds)
    };
    LowerSpectrum::Conditional {
        condition,
        family: family.map(Box::new),
    }
}

/// Constant `p`, `q`, `w`: `ν² = (q + λw)/p`.
pub(super) fn constant_coefficients(
    problem: &SturmLiouvilleProblem,
    p: ExprId,
    q: ExprId,
    w: ExprId,
    n: ExprId,
    pool: &ExprPool,
) -> Result<Eigensystem, PdeError> {
    let x = problem.var;
    let avoid = [p, q, w, problem.a, problem.b, x, n];
    let nu = fresh_symbol("nu", Domain::Positive, &avoid, pool);
    let kappa = fresh_symbol("kappa", Domain::Positive, &avoid, pool);
    let xa = simp(sub(x, problem.a, pool), pool);
    let arg = |f: ExprId| simp(pool.mul(vec![f, xa]), pool);
    let lam = simp(
        div(
            sub(pool.mul(vec![p, pool.pow(nu, pool.integer(2))]), q, pool),
            w,
            pool,
        ),
        pool,
    );
    let lam0 = simp(neg(div(q, w, pool), pool), pool);
    let lam_k = simp(
        div(
            neg(
                pool.add(vec![pool.mul(vec![p, pool.pow(kappa, pool.integer(2))]), q]),
                pool,
            ),
            w,
            pool,
        ),
        pool,
    );
    let basis = Basis {
        freq: nu,
        y: [
            pool.func("cos", vec![arg(nu)]),
            pool.func("sin", vec![arg(nu)]),
        ],
        lam,
        y0: [pool.integer(1), xa],
        lam0,
        lower: Some((
            kappa,
            [
                pool.func("cosh", vec![arg(kappa)]),
                pool.func("sinh", vec![arg(kappa)]),
            ],
            lam_k,
        )),
        kind: "regular_constant",
    };
    match problem.boundary {
        SlBoundary::Periodic => periodic(problem, p, q, w, basis, n, pool),
        SlBoundary::Separated { .. } => separated(problem, p, q, w, basis, n, pool),
    }
}

/// Periodic conditions with constant coefficients: `ν_n = 2nπ/ℓ`, two
/// eigenfunctions each, and the constant mode.
fn periodic(
    problem: &SturmLiouvilleProblem,
    p: ExprId,
    q: ExprId,
    w: ExprId,
    basis: Basis,
    n: ExprId,
    pool: &ExprPool,
) -> Result<Eigensystem, PdeError> {
    let x = problem.var;
    let mut side = Vec::new();
    require_positive(p, "p", problem, &mut side, pool)?;
    require_positive(w, "w", problem, &mut side, pool)?;
    let ell = simp(sub(problem.b, problem.a, pool), pool);
    let pi = pool.symbol("pi", Domain::Real);
    let nu_n = simp(
        pool.mul(vec![
            pool.integer(2),
            n,
            pi,
            pool.pow(ell, pool.integer(-1)),
        ]),
        pool,
    );
    let mut statuses = Vec::new();
    let mut funcs = Vec::new();
    let mut norms = Vec::new();
    for &yk in &basis.y {
        let ode = zstatus(ode_residual(x, p, q, w, basis.lam, yk, pool)?, pool);
        let at = |e: ExprId| simp(subs1(e, basis.freq, nu_n, pool), pool);
        let f = at(yk);
        let df = d(f, x, pool)?;
        let r1 = closed_status(
            sub(
                subs1(f, x, problem.a, pool),
                subs1(f, x, problem.b, pool),
                pool,
            ),
            pool,
        );
        let r2 = closed_status(
            sub(
                subs1(df, x, problem.a, pool),
                subs1(df, x, problem.b, pool),
                pool,
            ),
            pool,
        );
        statuses.extend([ode, r1, r2]);
        funcs.push(f);
        norms.push(
            crate::integrate::integrate_definite(
                pool.mul(vec![w, pool.pow(yk, pool.integer(2))]),
                x,
                problem.a,
                problem.b,
                pool,
            )
            .ok()
            .map(|r| reduce_pi_shifts(simp(subs1(r.value, basis.freq, nu_n, pool), pool), pool)),
        );
    }
    let status = combine(&statuses);
    if status == VerificationStatus::Refuted {
        return Err(PdeError::SlVerificationFailed(
            "periodic eigenfunction".into(),
        ));
    }
    let dfun = pool.func(
        "sin",
        vec![simp(
            pool.mul(vec![basis.freq, ell, pool.rational(1, 2)]),
            pool,
        )],
    );
    let fam = EigenFamily {
        index: n,
        first: 1,
        frequency_var: basis.freq,
        frequency: nu_n,
        eigenvalue: simp(subs1(basis.lam, basis.freq, nu_n, pool), pool),
        eigenvalue_of_frequency: basis.lam,
        eigenfunctions: funcs,
        eigenfunctions_of_frequency: basis.y.to_vec(),
        norms_squared: norms,
        closed_form: true,
        defining_equation: Some(dfun),
        enclosures: Vec::new(),
        verification: EigenVerification {
            status,
            ode: statuses[0],
            left: statuses[1],
            right: statuses[2],
            method: "closed_form",
            members: Vec::new(),
        },
    };
    let one = pool.integer(1);
    let ode0 = zstatus(ode_residual(x, p, q, w, basis.lam0, one, pool)?, pool);
    let mode = EigenMode {
        eigenvalue: basis.lam0,
        eigenfunction: one,
        norm_squared: Some(simp(pool.mul(vec![w, ell]), pool)),
        condition: None,
        verification: status_of(ode0),
    };
    Ok(Eigensystem {
        kind: "periodic",
        var: x,
        weight: w,
        interval: (problem.a, problem.b),
        threshold: basis.lam0,
        families: vec![fam],
        modes: vec![mode],
        lower: LowerSpectrum::None,
        side_conditions: side,
        notes: vec!["each eigenvalue λ_n, n ≥ 1, has the two eigenfunctions listed".into()],
    })
}

/// Variable coefficients: Euler–Cauchy in closed form, otherwise `dsolve`.
pub(super) fn variable_coefficients(
    problem: &SturmLiouvilleProblem,
    p: ExprId,
    q: ExprId,
    w: ExprId,
    n: ExprId,
    pool: &ExprPool,
) -> Result<Eigensystem, PdeError> {
    if matches!(problem.boundary, SlBoundary::Periodic) {
        return Err(PdeError::SlUnsupported(
            "periodic conditions are implemented for constant coefficients only".into(),
        ));
    }
    if let Some(basis) = euler_basis(problem, p, q, w, n, pool)? {
        return separated(problem, p, q, w, basis, n, pool);
    }
    let basis = dsolve_basis(problem, p, q, w, n, pool)?;
    separated(problem, p, q, w, basis, n, pool)
}

/// `x p'/p = b₀` and `x²(q + λw)/p = c₀ + c₁λ` constant: Euler–Cauchy,
/// `X = x^σ (A cos(ω ln(x/a)) + B sin(ω ln(x/a)))`, `σ = (1 − b₀)/2`,
/// `ω² = c₀ + c₁λ − σ²`.
fn euler_basis(
    problem: &SturmLiouvilleProblem,
    p: ExprId,
    q: ExprId,
    w: ExprId,
    n: ExprId,
    pool: &ExprPool,
) -> Result<Option<Basis>, PdeError> {
    let x = problem.var;
    let b0 = simp(div(pool.mul(vec![x, d(p, x, pool)?]), p, pool), pool);
    let c0 = simp(
        div(pool.mul(vec![pool.pow(x, pool.integer(2)), q]), p, pool),
        pool,
    );
    let c1 = simp(
        div(pool.mul(vec![pool.pow(x, pool.integer(2)), w]), p, pool),
        pool,
    );
    if [b0, c0, c1].iter().any(|&e| mentions_var(e, x, pool)) {
        return Ok(None);
    }
    // The interval must lie in x > 0 (ln(x/a)).
    if super::sign(problem.a, pool) != Sgn::Pos {
        return Ok(None);
    }
    let avoid = [p, q, w, problem.a, problem.b, x, n];
    let om = fresh_symbol("omega", Domain::Positive, &avoid, pool);
    let kappa = fresh_symbol("kappa", Domain::Positive, &avoid, pool);
    let sigma = simp(
        pool.mul(vec![sub(pool.integer(1), b0, pool), pool.rational(1, 2)]),
        pool,
    );
    let s2 = simp(pool.pow(sigma, pool.integer(2)), pool);
    let ln = pool.func("log", vec![div(x, problem.a, pool)]);
    let xs = simp(pool.pow(x, sigma), pool);
    let lam = simp(
        div(
            sub(pool.add(vec![pool.pow(om, pool.integer(2)), s2]), c0, pool),
            c1,
            pool,
        ),
        pool,
    );
    let lam0 = simp(div(sub(s2, c0, pool), c1, pool), pool);
    let lam_k = simp(
        div(
            sub(sub(s2, pool.pow(kappa, pool.integer(2)), pool), c0, pool),
            c1,
            pool,
        ),
        pool,
    );
    let arg = |f: ExprId| pool.mul(vec![f, ln]);
    Ok(Some(Basis {
        freq: om,
        y: [
            simp(pool.mul(vec![xs, pool.func("cos", vec![arg(om)])]), pool),
            simp(pool.mul(vec![xs, pool.func("sin", vec![arg(om)])]), pool),
        ],
        lam,
        y0: [xs, simp(pool.mul(vec![xs, ln]), pool)],
        lam0,
        lower: Some((
            kappa,
            [
                simp(
                    pool.mul(vec![xs, pool.func("cosh", vec![arg(kappa)])]),
                    pool,
                ),
                simp(
                    pool.mul(vec![xs, pool.func("sinh", vec![arg(kappa)])]),
                    pool,
                ),
            ],
            lam_k,
        )),
        kind: "regular",
    }))
}

/// `dsolve` on `p y'' + p' y' + (q + μ² w) y = 0`.
fn dsolve_basis(
    problem: &SturmLiouvilleProblem,
    p: ExprId,
    q: ExprId,
    w: ExprId,
    n: ExprId,
    pool: &ExprPool,
) -> Result<Basis, PdeError> {
    let x = problem.var;
    let avoid = [p, q, w, problem.a, problem.b, x, n];
    let mu = fresh_symbol("mu", Domain::Positive, &avoid, pool);
    let lam = simp(pool.pow(mu, pool.integer(2)), pool);
    let solve = |lam_e: ExprId| -> Result<[ExprId; 2], PdeError> {
        let y = fresh_symbol("y", Domain::Real, &avoid, pool);
        let (inp, yp, ypp) = crate::ode::dsolve::OdeInput::second_order(x, y, pool);
        let eq = pool.add(vec![
            pool.mul(vec![p, ypp]),
            pool.mul(vec![d(p, x, pool)?, yp]),
            pool.mul(vec![pool.add(vec![q, pool.mul(vec![lam_e, w])]), y]),
        ]);
        let res = crate::ode::dsolve::dsolve(&inp.with_equation(eq), pool).map_err(|e| {
            PdeError::SlUnsupported(format!("dsolve does not solve the eigenvalue ODE: {e}"))
        })?;
        let sol = &res.solutions[0];
        if sol.constants.len() != 2 {
            return Err(PdeError::SlUnsupported(
                "dsolve returned a solution without two constants".into(),
            ));
        }
        let (c1, c2) = (sol.constants[0], sol.constants[1]);
        let y1 = simp(d(sol.y_of_x, c1, pool)?, pool);
        let y2 = simp(d(sol.y_of_x, c2, pool)?, pool);
        let rest = sub(
            sol.y_of_x,
            pool.add(vec![pool.mul(vec![c1, y1]), pool.mul(vec![c2, y2])]),
            pool,
        );
        if !proved_zero(rest, pool) || mentions_var(y1, c2, pool) || mentions_var(y2, c1, pool) {
            return Err(PdeError::SlUnsupported(
                "dsolve's general solution is not linear in its constants".into(),
            ));
        }
        Ok([y1, y2])
    };
    let y = solve(lam)?;
    let y0 = solve(pool.integer(0))?;
    Ok(Basis {
        freq: mu,
        y,
        lam,
        y0,
        lam0: pool.integer(0),
        lower: None,
        kind: "regular",
    })
}
