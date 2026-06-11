//! First-order ODE classes for [`super::dsolve`].
//!
//! Strategy: most classes need the equation solved for `y'`.  We extract the
//! coefficient of `y'` (`A = ∂F/∂y'`) and the remainder (`B = F − A·y'`); when
//! `A` is free of `y'` the equation is linear in `y'` and `y' = −B/A`.  Clairaut
//! (nonlinear in `y'`) is handled directly on the equation.

use super::{
    contains, ddx, div, integrate_or_decline, is_zero, residual_is_zero, simp, sub, subs1,
    ConstGen, DsolveError, DsolveResult, DsolveSolution, OdeInput,
};
use crate::kernel::eval_const::try_expr_f64;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

pub(crate) fn solve(
    input: &OdeInput,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<DsolveResult, DsolveError> {
    let yp = input.derivs[0];

    // Clairaut is nonlinear in y'; try it before the linear-in-y' reduction.
    if let Some(res) = try_clairaut(input, gen, pool)? {
        return Ok(res);
    }

    // Solve for y': require equation linear in y'.
    let a = ddx(input.equation, yp, pool)?; // ∂F/∂y'
    if contains(a, yp, pool) || is_zero(a, pool) {
        return Err(DsolveError::Unsupported(
            "equation is not linear in y' (or independent of y')".to_string(),
        ));
    }
    let ayp = simp(pool.mul(vec![a, yp]), pool);
    let b = sub(input.equation, ayp, pool); // F − A·y'
    if contains(b, yp, pool) {
        return Err(DsolveError::Unsupported(
            "equation is not affine in y'".to_string(),
        ));
    }
    // y' = rhs(x, y) = −B/A
    let rhs = simp(div(pool.mul(vec![pool.integer(-1_i32), b]), a, pool), pool);

    // Try classes in order of specificity.  A class that recognises its form
    // but cannot close a required integral returns `Err`; we let the remaining
    // classes try, so a later class can still solve (e.g. a homogeneous-looking
    // equation whose `v` integral does not close may still be exact).
    type Attempt = fn(
        &OdeInput,
        ExprId,
        &mut ConstGen,
        &ExprPool,
    ) -> Result<Option<DsolveResult>, DsolveError>;
    let attempts: [Attempt; 6] = [
        try_separable,
        try_linear,
        try_bernoulli,
        try_exact,
        try_homogeneous,
        try_riccati,
    ];
    for attempt in attempts {
        if let Ok(Some(res)) = attempt(input, rhs, gen, pool) {
            return Ok(res);
        }
    }

    Err(DsolveError::Unsupported(
        "no implemented first-order class matched".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn finalize(
    input: &OdeInput,
    y_of_x: ExprId,
    constants: Vec<ExprId>,
    method: &'static str,
    pool: &ExprPool,
) -> Result<Option<DsolveResult>, DsolveError> {
    let y_of_x = simp(y_of_x, pool);
    match residual_is_zero(input, y_of_x, &constants, pool) {
        Ok(()) => Ok(Some(DsolveResult {
            solutions: vec![DsolveSolution {
                y_of_x,
                constants,
                method,
            }],
        })),
        // A candidate that fails verification means this class mis-fired; let
        // the dispatcher try the next class rather than aborting.
        Err(_) => Ok(None),
    }
}

/// Try to write `expr` as `g(x) · h(y)` (multiplicative split).  Returns
/// `(g_of_x, h_of_y)` when the factorisation is clean.
fn separable_split(
    expr: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    // Gather multiplicative factors.
    let factors: Vec<ExprId> = match pool.get(expr) {
        ExprData::Mul(args) => args,
        _ => vec![expr],
    };
    let mut gx: Vec<ExprId> = Vec::new();
    let mut hy: Vec<ExprId> = Vec::new();
    for f in factors {
        let has_x = contains(f, x, pool);
        let has_y = contains(f, y, pool);
        match (has_x, has_y) {
            (true, false) => gx.push(f),
            (false, true) => hy.push(f),
            (false, false) => gx.push(f), // constant → lump into g(x)
            (true, true) => return None,  // mixed factor → not separable this way
        }
    }
    let g = if gx.is_empty() {
        pool.integer(1_i32)
    } else {
        pool.mul(gx)
    };
    let h = if hy.is_empty() {
        pool.integer(1_i32)
    } else {
        pool.mul(hy)
    };
    Some((simp(g, pool), simp(h, pool)))
}

// ---------------------------------------------------------------------------
// Separable: y' = g(x)·h(y)  →  ∫ dy/h(y) = ∫ g(x) dx + C
// ---------------------------------------------------------------------------

fn try_separable(
    input: &OdeInput,
    rhs: ExprId,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<DsolveResult>, DsolveError> {
    let (x, y) = (input.x, input.y);
    let Some((g, h)) = separable_split(rhs, x, y, pool) else {
        return Ok(None);
    };
    // Need genuine y-dependence in h for "separable" to be meaningful & invertible.
    if !contains(h, y, pool) {
        return Ok(None); // pure y' = g(x): handled by linear path (q only)
    }
    // ∫ 1/h(y) dy = ∫ g(x) dx + C  →  solve for y if invertible.
    let inv_h = simp(pool.pow(h, pool.integer(-1_i32)), pool);
    let lhs_int = integrate_or_decline(inv_h, y, pool)?; // H(y)
    let rhs_int = integrate_or_decline(g, x, pool)?; // G(x)
    let c = gen.fresh(pool);

    // Try to solve H(y) = G(x) + C explicitly when H is invertible in closed form.
    // Common closed forms: H(y) = log(y) → y = C·e^{G}; H(y) = y → y = G + C; etc.
    let target = simp(pool.add(vec![rhs_int, c]), pool);
    if let Some(y_expr) = invert_simple(lhs_int, y, target, pool) {
        return finalize(input, y_expr, vec![c], "separable", pool);
    }
    Ok(None)
}

/// Invert `lhs(y) = target` for a few common closed forms, returning `y = …`.
fn invert_simple(lhs: ExprId, y: ExprId, target: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let lhs = simp(lhs, pool);
    // Case H(y) = y  → y = target
    if lhs == y {
        return Some(target);
    }
    // Case H(y) = c0 * y  (linear in y) → y = target / c0
    let dl = super::ddx(lhs, y, pool).ok()?;
    if !contains(dl, y, pool) && !is_zero(dl, pool) {
        // lhs = dl*y + const; recover constant
        let lin = simp(pool.mul(vec![dl, y]), pool);
        let cst = sub(lhs, lin, pool);
        if !contains(cst, y, pool) {
            // dl*y + cst = target → y = (target - cst)/dl
            return Some(div(sub(target, cst, pool), dl, pool));
        }
    }
    // Case H(y) = log(y) (or k*log(y)) → y = exp(target/k)
    if let ExprData::Func { name, args } = pool.get(lhs) {
        if name == "log" && args.len() == 1 && args[0] == y {
            return Some(simp(pool.func("exp", vec![target]), pool));
        }
    }
    // k*log(y): lhs = k*log(y)
    if let ExprData::Mul(args) = pool.get(lhs) {
        let mut k_factors = Vec::new();
        let mut logy = false;
        for a in &args {
            match pool.get(*a) {
                ExprData::Func { name, args: fa }
                    if name == "log" && fa.len() == 1 && fa[0] == y =>
                {
                    logy = true;
                }
                _ => k_factors.push(*a),
            }
        }
        if logy {
            let k = simp(pool.mul(k_factors), pool);
            if !contains(k, y, pool) {
                let arg = div(target, k, pool);
                return Some(simp(pool.func("exp", vec![arg]), pool));
            }
        }
    }
    // Case H(y) = Σ kᵢ·log(aᵢy + bᵢ): exponentiate to Π (aᵢy+bᵢ)^{kᵢ} = e^T and
    // solve when the result is linear in y (covers the logistic and similar).
    if let Some(y_expr) = invert_log_sum(lhs, y, target, pool) {
        return Some(y_expr);
    }
    None
}

/// Invert `Σ kᵢ·log(aᵢ·y + bᵢ) = target` for `kᵢ ∈ {+1, −1}`, when
/// exponentiating yields an equation linear in `y`.  Returns `y = …`.
fn invert_log_sum(lhs: ExprId, y: ExprId, target: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let terms: Vec<ExprId> = match pool.get(lhs) {
        ExprData::Add(args) => args,
        _ => vec![lhs],
    };
    // Each term must be ±log(linear_in_y); collect (sign, arg).
    let mut numer_args: Vec<ExprId> = Vec::new(); // exponent +1
    let mut denom_args: Vec<ExprId> = Vec::new(); // exponent −1
    for t in terms {
        let (sign, arg) = match pool.get(t) {
            ExprData::Func { name, args } if name == "log" && args.len() == 1 => (1, args[0]),
            ExprData::Mul(factors) => {
                // expect exactly one log factor and a numeric ±1 coefficient
                let mut logarg = None;
                let mut coeff = 1.0;
                for f in &factors {
                    match pool.get(*f) {
                        ExprData::Func { name, args } if name == "log" && args.len() == 1 => {
                            if logarg.is_some() {
                                return None;
                            }
                            logarg = Some(args[0]);
                        }
                        _ => {
                            coeff *= try_expr_f64(*f, pool)?;
                        }
                    }
                }
                let arg = logarg?;
                if (coeff - 1.0).abs() < 1e-9 {
                    (1, arg)
                } else if (coeff + 1.0).abs() < 1e-9 {
                    (-1, arg)
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        // arg must be linear (degree ≤ 1) in y.
        let darg = super::ddx(arg, y, pool).ok()?;
        if contains(darg, y, pool) {
            return None;
        }
        if sign == 1 {
            numer_args.push(arg);
        } else {
            denom_args.push(arg);
        }
    }
    // Π numer / Π denom = e^T  →  Π numer − e^T·Π denom = 0, linear in y.
    let et = simp(pool.func("exp", vec![target]), pool);
    let num = if numer_args.is_empty() {
        pool.integer(1_i32)
    } else {
        simp(pool.mul(numer_args), pool)
    };
    let den = if denom_args.is_empty() {
        pool.integer(1_i32)
    } else {
        simp(pool.mul(denom_args), pool)
    };
    // equation E(y) = num − e^T·den = 0
    let e_y = sub(num, simp(pool.mul(vec![et, den]), pool), pool);
    // Solve linear: E = b·y + a → y = −a/b.
    let b = super::ddx(e_y, y, pool).ok()?;
    if contains(b, y, pool) || is_zero(b, pool) {
        return None;
    }
    let by = simp(pool.mul(vec![b, y]), pool);
    let a = sub(e_y, by, pool);
    if contains(a, y, pool) {
        return None;
    }
    Some(div(
        simp(pool.mul(vec![pool.integer(-1_i32), a]), pool),
        b,
        pool,
    ))
}

// ---------------------------------------------------------------------------
// Linear: y' = q(x) − p(x)·y  →  y = e^{−∫p}(∫ e^{∫p} q dx + C)
// ---------------------------------------------------------------------------

/// Decompose `rhs` (= y') as `q(x) + r(x)·y` when affine in `y`.  Returns
/// `(p, q)` for the standard form `y' + p·y = q`, i.e. `p = −r`.
fn linear_split(rhs: ExprId, _x: ExprId, y: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let dr = super::ddx(rhs, y, pool).ok()?; // ∂rhs/∂y = r(x)
    if contains(dr, y, pool) {
        return None; // not affine in y
    }
    let ry = simp(pool.mul(vec![dr, y]), pool);
    let q = sub(rhs, ry, pool); // rhs − r·y = q(x)
    if contains(q, y, pool) {
        return None;
    }
    // p = −r (standard form y' + p·y = q)
    let p = simp(pool.mul(vec![pool.integer(-1_i32), dr]), pool);
    Some((p, q))
}

fn try_linear(
    input: &OdeInput,
    rhs: ExprId,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<DsolveResult>, DsolveError> {
    let (x, y) = (input.x, input.y);
    let Some((p, q)) = linear_split(rhs, x, y, pool) else {
        return Ok(None);
    };
    // Integrating factor μ = e^{∫p dx}
    let int_p = integrate_or_decline(p, x, pool)?;
    let mu = simp(pool.func("exp", vec![int_p]), pool);
    // y = (∫ μ q dx + C) / μ
    let muq = simp(pool.mul(vec![mu, q]), pool);
    let int_muq = integrate_or_decline(muq, x, pool)?;
    let c = gen.fresh(pool);
    let numer = simp(pool.add(vec![int_muq, c]), pool);
    let y_expr = div(numer, mu, pool);
    finalize(input, y_expr, vec![c], "linear", pool)
}

// ---------------------------------------------------------------------------
// Bernoulli: y' + p y = q y^n  (n ≠ 0,1)  →  v = y^{1−n} linearises
// ---------------------------------------------------------------------------

fn try_bernoulli(
    input: &OdeInput,
    rhs: ExprId,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<DsolveResult>, DsolveError> {
    let (x, y) = (input.x, input.y);
    // rhs = −p·y + q·y^n.  Try to detect: split additive terms by power of y.
    let terms: Vec<ExprId> = match pool.get(rhs) {
        ExprData::Add(args) => args,
        _ => vec![rhs],
    };
    // For each additive term, find its power of y (must be a clean monomial in y).
    let mut linear_coeff: Vec<ExprId> = Vec::new(); // coeff of y^1
    let mut bern_coeff: Vec<ExprId> = Vec::new(); // coeff of y^n
    let mut n_exp: Option<i64> = None;
    for t in terms {
        let Some((coeff, pw)) = monomial_in_y(t, y, pool) else {
            return Ok(None);
        };
        if contains(coeff, y, pool) {
            return Ok(None);
        }
        match pw {
            1 => linear_coeff.push(coeff),
            other => {
                if other == 0 {
                    // constant term → would be Bernoulli with q·y^0; treat n=0 not supported here
                    return Ok(None);
                }
                match n_exp {
                    None => n_exp = Some(other),
                    Some(e) if e == other => {}
                    Some(_) => return Ok(None), // two different nonlinear powers
                }
                bern_coeff.push(coeff);
            }
        }
    }
    let Some(n) = n_exp else {
        return Ok(None);
    };
    if n == 1 {
        return Ok(None);
    }
    let p = simp(
        pool.mul(vec![pool.integer(-1_i32), pool.add(linear_coeff)]),
        pool,
    ); // y' + p y, p = −(coeff of y)
    let q = simp(pool.add(bern_coeff), pool); // q·y^n
    if contains(q, y, pool) || contains(p, y, pool) {
        return Ok(None);
    }

    // v = y^{1−n}.  v' + (1−n) p v = (1−n) q.  Solve linear in v then y = v^{1/(1−n)}.
    let one_minus_n = pool.integer((1 - n) as i32);
    let pv = simp(pool.mul(vec![one_minus_n, p]), pool);
    let qv = simp(pool.mul(vec![one_minus_n, q]), pool);
    // integrating factor μ = e^{∫ pv}
    let int_pv = integrate_or_decline(pv, x, pool)?;
    let mu = simp(pool.func("exp", vec![int_pv]), pool);
    let muq = simp(pool.mul(vec![mu, qv]), pool);
    let int_muq = integrate_or_decline(muq, x, pool)?;
    let c = gen.fresh(pool);
    let v = div(simp(pool.add(vec![int_muq, c]), pool), mu, pool);
    // y = v^{1/(1−n)}
    let exp = pool.rational(1_i32, (1 - n) as i32);
    let y_expr = simp(pool.pow(v, exp), pool);
    finalize(input, y_expr, vec![c], "bernoulli", pool)
}

/// If `term` is `coeff · y^k` for integer `k ≥ 0` (with `coeff` free of `y`),
/// return `(coeff, k)`.  Returns `None` when `term` is not a clean integer-power
/// monomial in `y` (the caller treats that as "not this class").
fn monomial_in_y(term: ExprId, y: ExprId, pool: &ExprPool) -> Option<(ExprId, i64)> {
    // Decompose into factors, find the single y-power factor.
    let factors: Vec<ExprId> = match pool.get(term) {
        ExprData::Mul(args) => args,
        _ => vec![term],
    };
    let mut coeff: Vec<ExprId> = Vec::new();
    let mut power: i64 = 0;
    let mut found_y = false;
    for f in factors {
        if f == y {
            power += 1;
            found_y = true;
            continue;
        }
        if let ExprData::Pow { base, exp } = pool.get(f) {
            if base == y {
                if let ExprData::Integer(k) = pool.get(exp) {
                    power += k.0.to_i64()?;
                    found_y = true;
                    continue;
                }
                // non-integer power of y → not a clean monomial
                return None;
            }
        }
        if contains(f, y, pool) {
            // y appears inside a function/sub-expr → not a monomial
            return None;
        }
        coeff.push(f);
    }
    let c = if coeff.is_empty() {
        pool.integer(1_i32)
    } else {
        pool.mul(coeff)
    };
    Some((simp(c, pool), if found_y { power } else { 0 }))
}

// ---------------------------------------------------------------------------
// Homogeneous of degree zero: y' = G(y/x).  Substitute v = y/x.
// ---------------------------------------------------------------------------

fn try_homogeneous(
    input: &OdeInput,
    rhs: ExprId,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<DsolveResult>, DsolveError> {
    let (x, y) = (input.x, input.y);
    if !contains(rhs, x, pool) || !contains(rhs, y, pool) {
        return Ok(None);
    }
    // Replace y → v·x and check the result is free of x (degree-0 homogeneous).
    let v = pool.symbol("__v_hom", Domain::Real);
    let vx = simp(pool.mul(vec![v, x]), pool);
    let g_vx = subs1(rhs, y, vx, pool);
    if contains(g_vx, x, pool) {
        return Ok(None); // not homogeneous of degree zero
    }
    // Now y = v x, y' = v + x v'.  v + x v' = G(v)  →  x v' = G(v) − v.
    // Separable in v: ∫ dv/(G(v) − v) = ∫ dx/x = log|x| + C.
    let gmv = sub(g_vx, v, pool);
    if is_zero(gmv, pool) {
        return Ok(None); // y' = y/x → degenerate (linear), let linear handle it
    }
    let inv = simp(pool.pow(gmv, pool.integer(-1_i32)), pool);
    let lhs_int = integrate_or_decline(inv, v, pool)?; // Φ(v)
    let c = gen.fresh(pool);
    let logx = pool.func("log", vec![x]);
    let target = simp(pool.add(vec![logx, c]), pool);
    // Solve Φ(v) = log x + C for v, then y = v x.
    if let Some(v_expr) = invert_simple(lhs_int, v, target, pool) {
        let y_expr = simp(pool.mul(vec![v_expr, x]), pool);
        // back-substitute v occurrences (none should remain)
        if contains(y_expr, v, pool) {
            return Ok(None);
        }
        return finalize(input, y_expr, vec![c], "homogeneous", pool);
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Exact: M dx + N dy = 0 with ∂M/∂y = ∂N/∂x.  y' = −M/N.
// Solution F(x,y)=C with ∂F/∂x = M, ∂F/∂y = N.
// ---------------------------------------------------------------------------

fn try_exact(
    input: &OdeInput,
    _rhs: ExprId,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<DsolveResult>, DsolveError> {
    let (x, y, yp) = (input.x, input.y, input.derivs[0]);
    // Recover the natural M dx + N dy = 0 split from the *original* equation
    // (`equation = M + N·y'`):  N = ∂equation/∂y', M = equation − N·y'.
    let n = ddx(input.equation, yp, pool)?;
    if contains(n, yp, pool) || is_zero(n, pool) {
        return Ok(None);
    }
    let m = sub(input.equation, simp(pool.mul(vec![n, yp]), pool), pool);
    if contains(m, yp, pool) {
        return Ok(None);
    }
    // Exactness: ∂M/∂y == ∂N/∂x
    let dmy = ddx(m, y, pool)?;
    let dnx = ddx(n, x, pool)?;
    if !is_zero(sub(dmy, dnx, pool), pool) {
        return Ok(None);
    }
    // F = ∫ M dx + g(y), with ∂F/∂y = N → g'(y) = N − ∂/∂y ∫M dx.
    let int_m = integrate_or_decline(m, x, pool)?;
    let dint_m_dy = ddx(int_m, y, pool)?;
    let gy_prime = sub(n, dint_m_dy, pool); // should be free of x
    if contains(gy_prime, x, pool) {
        return Ok(None);
    }
    let g_of_y = integrate_or_decline(gy_prime, y, pool)?;
    let f = simp(pool.add(vec![int_m, g_of_y]), pool); // F(x,y)
                                                       // Implicit solution F(x,y) = C.  Try to solve for y if F is affine in y.
    let c = gen.fresh(pool);
    if let Some(y_expr) = solve_implicit_for_y(f, x, y, c, pool) {
        return finalize(input, y_expr, vec![c], "exact", pool);
    }
    Ok(None)
}

/// Solve `F(x,y) = C` for `y` when `F` is polynomial of degree ≤ 2 in `y`.
/// Affine `F = b·y + a` gives `y = (C − a)/b`; quadratic `F = A·y² + B·y + D`
/// gives one branch of the quadratic formula with `D` shifted by `−C`.
fn solve_implicit_for_y(
    f: ExprId,
    _x: ExprId,
    y: ExprId,
    c: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let b = super::ddx(f, y, pool).ok()?; // ∂F/∂y
    if !contains(b, y, pool) {
        // Affine case: F = b·y + a.
        if is_zero(b, pool) {
            return None;
        }
        let by = simp(pool.mul(vec![b, y]), pool);
        let a = sub(f, by, pool);
        if contains(a, y, pool) {
            return None;
        }
        return Some(div(sub(c, a, pool), b, pool));
    }
    // Quadratic case: ∂F/∂y = 2A·y + B  ⇒  A = ½·∂²F/∂y², B = ∂F/∂y|_{coeff}.
    let d2 = super::ddx(b, y, pool).ok()?; // ∂²F/∂y² = 2A
    if contains(d2, y, pool) || is_zero(d2, pool) {
        return None; // degree > 2 or not quadratic
    }
    let a_coeff = simp(pool.mul(vec![pool.rational(1_i32, 2_i32), d2]), pool); // A
                                                                               // B = (∂F/∂y) − 2A·y
    let b_coeff = sub(b, simp(pool.mul(vec![d2, y]), pool), pool);
    if contains(b_coeff, y, pool) {
        return None;
    }
    // D = F − A·y² − B·y
    let ay2 = simp(
        pool.mul(vec![a_coeff, pool.pow(y, pool.integer(2_i32))]),
        pool,
    );
    let by = simp(pool.mul(vec![b_coeff, y]), pool);
    let d_coeff = sub(sub(f, ay2, pool), by, pool); // free of y
    if contains(d_coeff, y, pool) {
        return None;
    }
    // A y² + B y + (D − C) = 0 → y = (−B + sqrt(B² − 4A(D−C)))/(2A)
    let dc = sub(d_coeff, c, pool);
    let disc = sub(
        simp(pool.pow(b_coeff, pool.integer(2_i32)), pool),
        simp(pool.mul(vec![pool.integer(4_i32), a_coeff, dc]), pool),
        pool,
    );
    let sqrt_disc = simp(pool.pow(disc, pool.rational(1_i32, 2_i32)), pool);
    let numer = simp(
        pool.add(vec![
            pool.mul(vec![pool.integer(-1_i32), b_coeff]),
            sqrt_disc,
        ]),
        pool,
    );
    let denom = simp(pool.mul(vec![pool.integer(2_i32), a_coeff]), pool);
    Some(div(numer, denom, pool))
}

// ---------------------------------------------------------------------------
// Riccati: y' = q0 + q1 y + q2 y², with a polynomial particular solution y_p.
// Then y = y_p + 1/v reduces to a linear ODE in v.  We decline if no
// low-degree polynomial particular solution exists.
// ---------------------------------------------------------------------------

fn try_riccati(
    input: &OdeInput,
    rhs: ExprId,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<DsolveResult>, DsolveError> {
    let (x, y) = (input.x, input.y);
    // rhs must be quadratic in y: rhs = q0 + q1 y + q2 y², q2 ≠ 0, qi free of y.
    let Some((_q0, q1, q2)) = quadratic_in_y(rhs, y, pool) else {
        return Ok(None);
    };
    if is_zero(q2, pool) {
        return Ok(None); // linear, not Riccati
    }
    // Find a polynomial particular solution y_p = a0 + a1 x + a2 x² (degree ≤ 2).
    let Some(yp_part) = find_poly_particular(rhs, x, y, pool) else {
        return Ok(None); // decline Riccati without a guessable particular solution
    };
    // y = y_p + 1/v.  v satisfies v' = −(q1 + 2 q2 y_p) v − q2  (linear).
    let two_q2_yp = simp(pool.mul(vec![pool.integer(2_i32), q2, yp_part]), pool);
    let p = simp(pool.add(vec![q1, two_q2_yp]), pool); // v' + p v = −q2
    let qrhs = simp(pool.mul(vec![pool.integer(-1_i32), q2]), pool);
    // linear in v: v' + p v = qrhs (here standard form v' + P v = Q with P=p, Q=qrhs)
    let int_p = integrate_or_decline(p, x, pool)?;
    let mu = simp(pool.func("exp", vec![int_p]), pool);
    let muq = simp(pool.mul(vec![mu, qrhs]), pool);
    let int_muq = integrate_or_decline(muq, x, pool)?;
    let c = gen.fresh(pool);
    let v = div(simp(pool.add(vec![int_muq, c]), pool), mu, pool);
    let inv_v = simp(pool.pow(v, pool.integer(-1_i32)), pool);
    let y_expr = simp(pool.add(vec![yp_part, inv_v]), pool);
    finalize(input, y_expr, vec![c], "riccati", pool)
}

/// Decompose `expr` as `q0 + q1·y + q2·y²` (each qi free of y).  Returns None if
/// it is not a polynomial of degree ≤ 2 in `y`.
fn quadratic_in_y(expr: ExprId, y: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId)> {
    let terms: Vec<ExprId> = match pool.get(expr) {
        ExprData::Add(args) => args,
        _ => vec![expr],
    };
    let mut q = [Vec::new(), Vec::new(), Vec::new()];
    for t in terms {
        let (coeff, pw) = monomial_in_y(t, y, pool)?;
        if contains(coeff, y, pool) || !(0..=2).contains(&pw) {
            return None;
        }
        q[pw as usize].push(coeff);
    }
    let mk = |v: Vec<ExprId>| {
        if v.is_empty() {
            pool.integer(0_i32)
        } else {
            simp(pool.add(v), pool)
        }
    };
    Some((mk(q[0].clone()), mk(q[1].clone()), mk(q[2].clone())))
}

/// Search for a polynomial particular solution `y_p = Σ aₖ xᵏ` (degree ≤ 2) of
/// `y' = rhs(x,y)`.  Uses a tiny rational-coefficient grid ansatz and the
/// numeric verification gate (sampling) — only constant/linear/quadratic are
/// attempted, which covers the standard textbook Riccati cases.
fn find_poly_particular(rhs: ExprId, x: ExprId, y: ExprId, pool: &ExprPool) -> Option<ExprId> {
    // Build candidate y_p, substitute into residual R = y_p' − rhs(x, y_p), and
    // require R ≡ 0 symbolically.  We solve for the coefficients by sampling.
    //
    // Try degrees 0,1,2.  For each, set up y_p with unknown rational coeffs and
    // solve the resulting polynomial-identity-in-x system over a small grid of
    // candidate integer/rational values (covers e.g. y_p = x, y_p = 1, y_p = -1/x
    // is excluded since we only do polynomials).
    let candidates = candidate_constants();
    for degree in 0..=2usize {
        for combo in coefficient_combinations(degree + 1, &candidates) {
            let yp = build_poly(&combo, x, pool);
            let ypp = super::ddx(yp, x, pool).ok()?;
            let r = subs1(rhs, y, yp, pool);
            let resid = sub(ypp, r, pool);
            if is_zero(resid, pool) {
                return Some(yp);
            }
        }
    }
    None
}

fn candidate_constants() -> Vec<(i64, i64)> {
    // (numerator, denominator) rationals to try for each coefficient.
    vec![
        (0, 1),
        (1, 1),
        (-1, 1),
        (2, 1),
        (-2, 1),
        (1, 2),
        (-1, 2),
        (3, 1),
        (-3, 1),
    ]
}

fn coefficient_combinations(n: usize, candidates: &[(i64, i64)]) -> Vec<Vec<(i64, i64)>> {
    if n == 0 {
        return vec![vec![]];
    }
    let rest = coefficient_combinations(n - 1, candidates);
    let mut out = Vec::new();
    for &c in candidates {
        for r in &rest {
            let mut v = vec![c];
            v.extend_from_slice(r);
            out.push(v);
        }
    }
    out
}

fn build_poly(coeffs: &[(i64, i64)], x: ExprId, pool: &ExprPool) -> ExprId {
    let mut terms = Vec::new();
    for (k, &(num, den)) in coeffs.iter().enumerate() {
        if num == 0 {
            continue;
        }
        let c = pool.rational(num, den);
        let term = if k == 0 {
            c
        } else {
            let xk = pool.pow(x, pool.integer(k as i32));
            pool.mul(vec![c, xk])
        };
        terms.push(term);
    }
    if terms.is_empty() {
        pool.integer(0_i32)
    } else {
        simp(pool.add(terms), pool)
    }
}

// ---------------------------------------------------------------------------
// Clairaut: y = x y' + f(y').  General solution y = C x + f(C).
// ---------------------------------------------------------------------------

fn try_clairaut(
    input: &OdeInput,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<DsolveResult>, DsolveError> {
    let (x, y, yp) = (input.x, input.y, input.derivs[0]);
    // Equation in the form y − x y' − f(y') = 0, i.e. equation = y − x·y' − f(y').
    // Detect: equation linear in y with coefficient 1 (or −1), and the
    // remaining part is −x·y' − f(y') (free of y).
    let coeff_y = ddx(input.equation, y, pool)?;
    // Require ∂/∂y = constant ±1
    let cy = try_expr_f64(simp(coeff_y, pool), pool);
    if cy != Some(1.0) && cy != Some(-1.0) {
        return Ok(None);
    }
    let sign = cy.unwrap();
    // rest = equation − coeff_y·y  (should be free of y)
    let rest = sub(input.equation, simp(pool.mul(vec![coeff_y, y]), pool), pool);
    if contains(rest, y, pool) {
        return Ok(None);
    }
    // Normalise to y = x y' + f(y'):  y = −rest/sign − ... actually
    // equation = sign·y + rest = 0 → y = −rest/sign.
    let y_solved = div(
        simp(pool.mul(vec![pool.integer(-1_i32), rest]), pool),
        simp(pool.integer(sign as i32), pool),
        pool,
    ); // should equal x·y' + f(y')
       // Check the Clairaut shape: y_solved − x·y' = f(y') must be free of x and y
       // (i.e. depend only on y'); the x·y' term carries all the x-dependence.
    let f_of_yp = sub(y_solved, simp(pool.mul(vec![x, yp]), pool), pool);
    if contains(f_of_yp, x, pool) || contains(f_of_yp, y, pool) {
        return Ok(None);
    }
    // Ensure y' genuinely appears as the bare x·y' term (reject e.g. y = x·(y')²
    // which is not Clairaut): y_solved must still contain y' linearly via x·y'.
    if !contains(y_solved, yp, pool) {
        return Ok(None);
    }
    // General solution: y = C x + f(C).
    let c = gen.fresh(pool);
    let f_of_c = subs1(f_of_yp, yp, c, pool);
    let y_expr = simp(pool.add(vec![pool.mul(vec![c, x]), f_of_c]), pool);
    finalize(input, y_expr, vec![c], "clairaut", pool)
}
