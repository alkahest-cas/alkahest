//! Linear ODE classes for [`super::dsolve`]: constant-coefficient (any order),
//! Euler–Cauchy, and general variable-coefficient second order.
//!
//! Coefficients of `y^(k)` are extracted from the equation.  When all are
//! constant *and numeric* the characteristic polynomial is built and factored
//! over ℚ (rational roots + quadratic factors); irreducible factors of degree
//! ≥ 3 are declined.  When they are constant but **symbolic** — `y'' + 2ζω y' +
//! ω² y = 0`, the damped oscillator — the characteristic polynomial is rooted
//! in closed form up to degree two after a `λᵏ` factor is peeled off, and the
//! branch its discriminant leaves open is reported rather than assumed (see
//! [`symbolic_basis`]).  Euler–Cauchy `a·x²y'' + b·x·y' + c·y = r` is detected
//! by the `xᵏ` coefficient pattern.  Anything else at second order falls
//! through to reduction of order.
//!
//! All three routes end at the same place: a fundamental system, handed to
//! [`super::variation`] for the forcing term.  Only the way the basis is found
//! differs.

use super::{
    contains, ddx, exact_roots, is_zero, residual_is_zero, simp, sub, AssumedSign, ConstGen,
    DsolveBranch, DsolveError, OdeInput, SolveCtx,
};
use crate::kernel::eval_const::try_expr_f64;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::matrix::zero_test::{zero_status, ZeroStatus};
use rug::Rational;

// ---------------------------------------------------------------------------
// Coefficient extraction
// ---------------------------------------------------------------------------

/// Extract the coefficient of `y^(k)` for `k = 0..=n` and the inhomogeneous
/// RHS `r(x)`.  Equation is `Σ aₖ y^(k) − r(x) = 0`, so we collect coefficients
/// linear in each derivative symbol and the constant-in-derivatives remainder
/// becomes `−r`.  Returns `(coeffs, r)` with `coeffs[k]` the coefficient of the
/// k-th derivative (k=0 is `y`).  All coefficients must be free of `y` and all
/// derivative symbols (linearity check).
fn extract_linear(input: &OdeInput, pool: &ExprPool) -> Result<(Vec<ExprId>, ExprId), DsolveError> {
    let n = input.order();
    let mut coeffs = Vec::with_capacity(n + 1);
    // coefficient of y (0-th derivative)
    let mut deriv_syms = vec![input.y];
    deriv_syms.extend_from_slice(&input.derivs);

    // For each derivative symbol s, coeff = ∂equation/∂s; must be free of all syms.
    for &s in &deriv_syms {
        let c = ddx(input.equation, s, pool)?;
        for &other in &deriv_syms {
            if contains(c, other, pool) {
                return Err(DsolveError::Unsupported(
                    "equation is not linear in y and its derivatives".to_string(),
                ));
            }
        }
        coeffs.push(simp(c, pool));
    }
    // remainder r0 = equation − Σ coeff_k · sym_k  (free of all derivative syms)
    let mut acc = input.equation;
    for (c, &s) in coeffs.iter().zip(deriv_syms.iter()) {
        let term = simp(pool.mul(vec![*c, s]), pool);
        acc = sub(acc, term, pool);
    }
    // acc = −r(x)  →  r = −acc
    for &s in &deriv_syms {
        if contains(acc, s, pool) {
            return Err(DsolveError::Unsupported(
                "equation is not affine in derivatives".to_string(),
            ));
        }
    }
    let r = simp(pool.mul(vec![pool.integer(-1_i32), acc]), pool);
    Ok((coeffs, r))
}

fn all_constant(coeffs: &[ExprId], x: ExprId, pool: &ExprPool) -> bool {
    coeffs.iter().all(|&c| !contains(c, x, pool))
}

// ---------------------------------------------------------------------------
// Second-order dispatch
// ---------------------------------------------------------------------------

pub(crate) fn solve_second_order(
    input: &OdeInput,
    gen: &mut ConstGen,
    ctx: &mut SolveCtx<'_>,
    pool: &ExprPool,
) -> Result<Vec<DsolveBranch>, DsolveError> {
    let (coeffs, r) = match extract_linear(input, pool) {
        Ok(v) => v,
        Err(e) => {
            // Nonlinear, but free of `y`: `p = y'` makes it first order.
            if let Some(res) = super::special_linear::solve_y_absent(input, gen, pool)? {
                return Ok(res);
            }
            // `y'' = f(y)` with an elliptic energy integral, or Liouville's
            // `y'' = a·e^{by}`: the classical inversions, before the `p(y)`
            // route below can settle for an implicit quadrature.
            if let Some(res) = super::autonomous::solve_autonomous_special(input, gen, pool)? {
                return Ok(res);
            }
            // Nonlinear and free of `x`: `p(y) = y'` makes it first order.
            if let Some(res) = super::special_linear::solve_x_absent(input, gen, pool)? {
                return Ok(res);
            }
            return Err(e);
        }
    };
    let x = input.x;

    if all_constant(&coeffs, x, pool) {
        return solve_const_coeff(input, &coeffs, r, gen, ctx, pool);
    }
    // Euler–Cauchy: coeffs[k] = c_k · x^k with c_k constant.
    if let Some(euler) = try_euler_cauchy(input, &coeffs, r, gen, pool)? {
        return Ok(euler);
    }
    // Airy (`y'' = (a x + b) y`, after removing a constant first-derivative
    // term) and the Bessel family `x^α Z_ν(β x^γ)`.
    if let Some(basis) = super::special_linear::airy_basis(&coeffs, x, pool) {
        let mark = gen.checkpoint();
        if let Some(res) = compose_from_basis(input, &basis, coeffs[2], r, "airy", gen, pool)? {
            return Ok(res);
        }
        gen.rollback(mark, pool);
    }
    if let Some(basis) = super::special_linear::bessel_basis(&coeffs, x, pool) {
        let mark = gen.checkpoint();
        if let Some(res) = compose_from_basis(input, &basis, coeffs[2], r, "bessel", gen, pool)? {
            return Ok(res);
        }
        gen.rollback(mark, pool);
    }
    // General variable coefficients: one solution by ansatz, the second by
    // reduction of order, the forcing by variation of parameters.
    if let Some(res) = try_reduction_of_order(input, &coeffs, r, gen, pool)? {
        return Ok(res);
    }
    Err(DsolveError::Unsupported(
        "second-order equation is neither constant-coefficient nor Euler–Cauchy, \
         and no first solution was found for reduction of order"
            .to_string(),
    ))
}

// ---------------------------------------------------------------------------
// General variable-coefficient second order, via reduction of order
// ---------------------------------------------------------------------------

/// `a₂(x) y'' + a₁(x) y' + a₀(x) y = r(x)` with arbitrary coefficient
/// functions.
///
/// Normalise to `y'' + P y' + Q y = g`, find one homogeneous solution `y₁`
/// from a small ansatz list, get `y₂ = y₁∫e^{−∫P}/y₁²` by reduction of order,
/// and close the forcing with variation of parameters.  Everything after the
/// ansatz is quadrature, which is why this class only became reachable once
/// the integrands stopped arriving mangled.
fn try_reduction_of_order(
    input: &OdeInput,
    coeffs: &[ExprId],
    r: ExprId,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<Vec<DsolveBranch>>, DsolveError> {
    let x = input.x;
    if coeffs.len() != 3 || is_zero(coeffs[2], pool) {
        return Ok(None);
    }
    let a2 = coeffs[2];
    let p = super::normalized(super::div(coeffs[1], a2, pool), pool);
    let q = super::normalized(super::div(coeffs[0], a2, pool), pool);
    // Polynomial solutions first (Legendre, Hermite, Laguerre, Chebyshev at
    // integer parameter): two of them are a fundamental system outright.
    let polys = exact_roots::polynomial_solutions(coeffs, x, pool);
    if polys.len() >= 2 && independent(polys[0], polys[1], x, pool) {
        return compose_from_basis(input, &polys[..2], a2, r, "polynomial_solutions", gen, pool);
    }
    let y1 = match polys.first() {
        Some(&y1) => y1,
        None => match super::variation::find_first_solution(p, q, x, pool)? {
            Some(y1) => y1,
            None => return Ok(None),
        },
    };
    if let Some(y2) = super::variation::reduction_of_order(y1, p, x, pool)? {
        let mark = gen.checkpoint();
        if let Some(res) =
            compose_from_basis(input, &[y1, y2], a2, r, "reduction_of_order", gen, pool)?
        {
            return Ok(Some(res));
        }
        gen.rollback(mark, pool);
    }
    // The quadrature for y₂ did not close: look for y₂ = w(x)·(polynomial)
    // for a few weights whose logarithmic derivative is rational — `√a₂`
    // gives Chebyshev's `√(1 − x²)·U_{n−1}`.
    let Some(y2) = weighted_polynomial_solution(coeffs, y1, x, pool)? else {
        return Ok(None);
    };
    compose_from_basis(
        input,
        &[y1, y2],
        a2,
        r,
        "weighted_polynomial_solution",
        gen,
        pool,
    )
}

/// A solution `w·u` of `a₂y'' + a₁y' + a₀y = 0`, independent of `y1`, with
/// `u` a polynomial and `w` one of `a₂^{±1/2}`, `e^{±x}`, `e^{±x²}`,
/// `e^{±x²/2}`. For `y = w·u` the equation for `u` is
/// `a₂u'' + (a₁ + 2a₂g₁)u' + (a₀ + a₁g₁ + a₂g₂)u = 0` with `g₁ = w'/w`,
/// `g₂ = w''/w`, which has rational coefficients exactly when `g₁` does.
fn weighted_polynomial_solution(
    coeffs: &[ExprId],
    y1: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, DsolveError> {
    let (a0, a1, a2) = (coeffs[0], coeffs[1], coeffs[2]);
    let half = pool.rational(1, 2);
    let x2 = pool.pow(x, pool.integer(2_i32));
    // (w, g₁ = w'/w, g₂ = w''/w), written out so that g₁, g₂ are visibly
    // rational: for w = a₂^k, g₁ = k·a₂'/a₂ and g₂ = k(k−1)(a₂'/a₂)² + k·a₂''/a₂;
    // for w = e^g, g₁ = g' and g₂ = g'' + g'².
    let mut weights: Vec<(ExprId, ExprId, ExprId)> = Vec::new();
    let da2 = ddx(a2, x, pool)?;
    let dda2 = ddx(da2, x, pool)?;
    if contains(a2, x, pool) {
        let l = super::div(da2, a2, pool);
        let l2 = super::div(dda2, a2, pool);
        for k in [pool.rational(1, 2), pool.rational(-1, 2)] {
            let km1 = pool.add(vec![k, pool.integer(-1_i32)]);
            weights.push((
                pool.pow(a2, k),
                pool.mul(vec![k, l]),
                pool.add(vec![
                    pool.mul(vec![k, km1, pool.pow(l, pool.integer(2_i32))]),
                    pool.mul(vec![k, l2]),
                ]),
            ));
        }
    }
    for g in [
        x,
        pool.mul(vec![pool.integer(-1_i32), x]),
        x2,
        pool.mul(vec![pool.integer(-1_i32), x2]),
        pool.mul(vec![half, x2]),
        pool.mul(vec![pool.rational(-1, 2), x2]),
    ] {
        let g1 = ddx(g, x, pool)?;
        let g2 = ddx(g1, x, pool)?;
        weights.push((
            pool.func("exp", vec![g]),
            g1,
            pool.add(vec![g2, pool.pow(g1, pool.integer(2_i32))]),
        ));
    }
    for (w, g1, g2) in weights {
        let w = simp(w, pool);
        let b1 = simp(
            pool.add(vec![a1, pool.mul(vec![pool.integer(2_i32), a2, g1])]),
            pool,
        );
        let b0 = simp(
            pool.add(vec![a0, pool.mul(vec![a1, g1]), pool.mul(vec![a2, g2])]),
            pool,
        );
        for u in exact_roots::polynomial_solutions(&[b0, b1, a2], x, pool) {
            let y2 = simp(pool.mul(vec![w, u]), pool);
            if independent(y1, y2, x, pool) {
                return Ok(Some(y2));
            }
        }
    }
    Ok(None)
}

/// The Wronskian `y1·y2' − y1'·y2` is non-zero at some sample point.
fn independent(y1: ExprId, y2: ExprId, x: ExprId, pool: &ExprPool) -> bool {
    let (Ok(d1), Ok(d2)) = (ddx(y1, x, pool), ddx(y2, x, pool)) else {
        return false;
    };
    let w = pool.add(vec![
        pool.mul(vec![y1, d2]),
        pool.mul(vec![pool.integer(-1_i32), d1, y2]),
    ]);
    [0.31_f64, 0.57, 0.83].iter().any(|&xv| {
        let env = std::collections::HashMap::from([(x, xv)]);
        super::verify::eval(w, &env, pool).is_some_and(|v| v.is_finite() && v.abs() > 1e-9)
    })
}

// ---------------------------------------------------------------------------
// Constant-coefficient (any order ≥ 1): homogeneous via characteristic poly
// ---------------------------------------------------------------------------

pub(crate) fn solve_higher_order(
    input: &OdeInput,
    _n: usize,
    gen: &mut ConstGen,
    ctx: &mut SolveCtx<'_>,
    pool: &ExprPool,
) -> Result<Vec<DsolveBranch>, DsolveError> {
    let (coeffs, r) = extract_linear(input, pool)?;
    let x = input.x;
    if !all_constant(&coeffs, x, pool) {
        if let Some(euler) = try_euler_cauchy(input, &coeffs, r, gen, pool)? {
            return Ok(euler);
        }
        return Err(DsolveError::Unsupported(
            "higher-order non-constant-coefficient equations are supported only in \
             Euler–Cauchy form (coefficient of y^(k) a rational multiple of x^k)"
                .to_string(),
        ));
    }
    solve_const_coeff(input, &coeffs, r, gen, ctx, pool)
}

fn solve_const_coeff(
    input: &OdeInput,
    coeffs: &[ExprId],
    r: ExprId,
    gen: &mut ConstGen,
    ctx: &mut SolveCtx<'_>,
    pool: &ExprPool,
) -> Result<Vec<DsolveBranch>, DsolveError> {
    let x = input.x;
    // Characteristic polynomial Σ aₖ λᵏ.  Rational coefficients take the exact
    // route (FLINT factorisation over ℚ, roots in radicals or `RootOf`); a
    // coefficient that is not a rational literal sends the equation down the
    // closed-form symbolic route instead.  No root is ever a float.
    let rational = rational_coeffs(coeffs, pool);
    let (basis, method) = match &rational {
        Some(a) => {
            let roots = exact_roots::exact_roots(a, pool)?;
            (
                basis_from_roots(&roots, x, false, pool),
                "constant_coefficient",
            )
        }
        None => symbolic_basis(coeffs, x, ctx, pool)?,
    };

    let mut terms = Vec::new();
    let mut constants = Vec::new();
    for b in &basis {
        let c = gen.fresh(pool);
        constants.push(c);
        terms.push(pool.mul(vec![c, *b]));
    }
    let y_homog = simp(pool.add(terms), pool);

    // Non-homogeneous: find a particular solution.
    let y_general = if is_zero(r, pool) {
        y_homog
    } else {
        let yp = particular_solution(input, coeffs, rational.as_deref(), &basis, r, pool)?;
        simp(pool.add(vec![y_homog, yp]), pool)
    };

    match residual_is_zero(input, y_general, &constants, pool) {
        Ok(()) => Ok(vec![DsolveBranch::explicit(y_general, constants, method)]),
        Err(e) => Err(e),
    }
}

/// The coefficients as rational literals, leading zeros dropped; `None` when
/// one of them is not a rational literal.
fn rational_coeffs(coeffs: &[ExprId], pool: &ExprPool) -> Option<Vec<Rational>> {
    let mut a: Vec<Rational> = coeffs
        .iter()
        .map(|&c| exact_roots::literal(simp(c, pool), pool))
        .collect::<Option<_>>()?;
    while a.len() > 1 && a.last().is_some_and(|c| *c == 0) {
        a.pop();
    }
    Some(a)
}

/// Fundamental system from exact roots. `euler = false`: `xʲe^{rx}`,
/// `xʲe^{αx}cos βx`, `xʲe^{αx}sin βx`. `euler = true` (roots of the indicial
/// polynomial): `x^r logʲx`, `x^α cos(β log x) logʲx`, `x^α sin(β log x) logʲx`.
fn basis_from_roots(
    roots: &[exact_roots::ExactRoot],
    x: ExprId,
    euler: bool,
    pool: &ExprPool,
) -> Vec<ExprId> {
    use exact_roots::ExactRoot;
    let logx = pool.func("log", vec![x]);
    let t = if euler { logx } else { x };
    let growth = |r: ExprId| -> ExprId {
        if matches!(zero_status(pool, r), ZeroStatus::Zero) {
            pool.integer(1_i32)
        } else if euler {
            pool.pow(x, r)
        } else {
            pool.func("exp", vec![simp(pool.mul(vec![r, x]), pool)])
        }
    };
    let power = |j: usize| -> ExprId {
        if j == 0 {
            pool.integer(1_i32)
        } else {
            pool.pow(t, pool.integer(j as i64))
        }
    };
    let mut out = Vec::new();
    for root in roots {
        match *root {
            ExactRoot::Real { r, m } => {
                let g = growth(r);
                for j in 0..m {
                    out.push(simp(pool.mul(vec![power(j), g]), pool));
                }
            }
            ExactRoot::Complex { alpha, beta, m } => {
                let g = growth(alpha);
                let arg = simp(pool.mul(vec![beta, t]), pool);
                for j in 0..m {
                    for trig in ["cos", "sin"] {
                        out.push(simp(
                            pool.mul(vec![power(j), g, pool.func(trig, vec![arg])]),
                            pool,
                        ));
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Constant-coefficient with *symbolic* coefficients
// ---------------------------------------------------------------------------

/// Fundamental system of `Σ aₖ y^(k) = 0` when the `aₖ` are constant in `x` but
/// contain free symbols, together with the method label naming the branch taken.
///
/// # The branch problem
///
/// For symbolic coefficients the multiplicity structure of the characteristic
/// polynomial is *parameter dependent*, so it is not decidable here: the
/// discriminant of `λ² + 2ζω λ + ω²` is `4ω²(ζ² − 1)`, which is positive,
/// zero and negative on three non-empty regions of the `(ζ, ω)` plane.  The
/// four ways out, in the order they are tried:
///
/// 1. **Proof.** `zero_test::zero_status` decides the discriminant outright
///    for the cases that are decidable (`λ² + 2aλ + a²` really does have a
///    double root for every `a`).  No condition is attached.
/// 2. **Assumptions.** A caller-stated `D > 0`, `D ≠ 0` or `−D > 0` settles it.
///    `−D > 0` additionally selects the real oscillatory form
///    `e^{αx}(C₁cos βx + C₂sin βx)` over the complex exponentials.
/// 3. **A side condition.** Otherwise the uniform two-exponential form is
///    returned — it satisfies the equation identically, for *every* parameter
///    value, and it is what SymPy and most CAS return — with `D ≠ 0` recorded
///    as a [`SideCondition`](crate::deriv::SideCondition) and a note naming the
///    confluent case it does not span.  Silently assuming distinct roots is
///    what this exists to prevent.
/// 4. **Refusal.** Degree ≥ 3 after peeling the `λᵏ` factor has no closed-form
///    root in general, and is declined with a message naming the degree.
fn symbolic_basis(
    coeffs: &[ExprId],
    x: ExprId,
    ctx: &mut SolveCtx<'_>,
    pool: &ExprPool,
) -> Result<(Vec<ExprId>, &'static str), DsolveError> {
    // `a[k]` is the coefficient of λᵏ.  Drop provably-zero leading terms; a
    // leading coefficient that is merely *not provably* zero stays, and the
    // order it claims becomes a side condition.
    let mut a: Vec<ExprId> = coeffs.iter().map(|&c| simp(c, pool)).collect();
    while a.len() > 1 && matches!(zero_status(pool, *a.last().unwrap()), ZeroStatus::Zero) {
        a.pop();
    }
    let deg = a.len() - 1;
    if deg == 0 {
        return Err(DsolveError::Unsupported(
            "every derivative coefficient vanishes; this is an algebraic equation, not an ODE"
                .to_string(),
        ));
    }
    require_nonzero(a[deg], ctx, pool);

    // Peel the `λᵏ` factor: `a₀ = … = a_{k−1} = 0` contributes the root 0 with
    // multiplicity k, whose basis is `1, x, …, x^{k−1}`.
    let mut k = 0usize;
    while k < deg && matches!(zero_status(pool, a[k]), ZeroStatus::Zero) {
        k += 1;
    }
    let mut basis: Vec<ExprId> = (0..k)
        .map(|j| {
            if j == 0 {
                pool.integer(1_i32)
            } else {
                simp(pool.pow(x, pool.integer(j as i32)), pool)
            }
        })
        .collect();
    let b = &a[k..];

    let (rest, method) = match b.len() - 1 {
        0 => (Vec::new(), "constant_coefficient_symbolic"),
        1 => {
            // b₁λ + b₀ = 0 → λ = −b₀/b₁.
            require_nonzero(b[1], ctx, pool);
            let root = super::div(pool.mul(vec![pool.integer(-1_i32), b[0]]), b[1], pool);
            (
                vec![exp_of_root(root, x, pool)],
                "constant_coefficient_symbolic",
            )
        }
        2 => symbolic_quadratic(b[2], b[1], b[0], x, ctx, pool)?,
        d => {
            return Err(DsolveError::Unsupported(format!(
                "characteristic polynomial has symbolic coefficients and degree {d} after \
                 factoring out λ^{k}; no closed-form roots exist beyond degree 2 for \
                 arbitrary symbolic coefficients, and this solver will not return an \
                 unverifiable root form"
            )))
        }
    };
    basis.extend(rest);
    Ok((basis, method))
}

/// Fundamental system of `aλ² + bλ + c = 0`, branching on the discriminant.
fn symbolic_quadratic(
    a: ExprId,
    b: ExprId,
    c: ExprId,
    x: ExprId,
    ctx: &mut SolveCtx<'_>,
    pool: &ExprPool,
) -> Result<(Vec<ExprId>, &'static str), DsolveError> {
    let neg_one = pool.integer(-1_i32);
    let two_a = simp(pool.mul(vec![pool.integer(2_i32), a]), pool);
    // D = b² − 4ac, with integer powers of products distributed so the
    // vanishing test and the assumption matcher see one spelling.
    let disc = super::expand_powers(
        pool.add(vec![
            pool.pow(b, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(-4_i32), a, c]),
        ]),
        pool,
    );
    let neg_disc = simp(pool.mul(vec![neg_one, disc]), pool);
    // α = −b/(2a), the real part shared by every branch.
    let alpha = super::div(pool.mul(vec![neg_one, b]), two_a, pool);

    // 1. Decided outright: a genuine double root, for every parameter value.
    if matches!(zero_status(pool, disc), ZeroStatus::Zero) {
        let e = exp_of_root(alpha, x, pool);
        let xe = simp(pool.mul(vec![x, e]), pool);
        return Ok((vec![e, xe], "constant_coefficient_symbolic_repeated_root"));
    }

    // 2. Decided by the caller (or by the discriminant being a literal).
    let mut sign = match try_expr_f64(disc, pool) {
        Some(v) if v > 0.0 => AssumedSign::Positive,
        Some(v) if v < 0.0 => AssumedSign::Negative,
        _ => ctx.assumed_sign(disc, pool),
    };
    // 2b. Decided by the real-sign prover: symbol domains plus the stated
    //     assumptions, through sign algebra and a polynomial certificate —
    //     `−4k²` under `k > 0`, `4ω²(ζ²−1)` under `0 < ζ < 1`. A discriminant
    //     proved `≤ 0` but not `< 0` (`−4k²` for a real `k`) is negative
    //     wherever the `D ≠ 0` side condition below holds, so it selects the
    //     real oscillatory form under that same condition.
    let mut nonpositive_only = false;
    if matches!(sign, AssumedSign::Unknown | AssumedSign::NonZero) {
        let stated_nonzero = sign == AssumedSign::NonZero;
        let set = {
            let _scope = (!ctx.assumptions.predicates().is_empty())
                .then(|| crate::simplify::enter_assumptions(ctx.assumptions, pool));
            crate::simplify::real_form::sign_set(disc, pool)
        };
        match set {
            Some(s) if s.is_neg() => sign = AssumedSign::Negative,
            Some(s) if s.is_pos() => sign = AssumedSign::Positive,
            Some(s) if s.is_nonpos() && stated_nonzero => sign = AssumedSign::Negative,
            Some(s) if s.is_nonneg() && stated_nonzero => sign = AssumedSign::Positive,
            Some(s) if s.is_nonpos() => nonpositive_only = true,
            _ => {}
        }
    }
    let asserted_nonzero = sign != AssumedSign::Unknown || super::is_settled_nonzero(disc, pool);
    if sign == AssumedSign::Negative || nonpositive_only {
        if sign != AssumedSign::Negative && !asserted_nonzero {
            ctx.conds.require_nonzero(disc);
            ctx.conds.note(format!(
                "the characteristic discriminant {d} is never positive; where it vanishes the \
                 roots coincide and the general solution is C1·e^({r}·{v}) + C2·{v}·e^({r}·{v}), \
                 which this oscillatory family does not span",
                d = pool.display(disc),
                r = pool.display(alpha),
                v = pool.display(x),
            ));
        }
        // Negative discriminant: the real oscillatory form.  β = √(−D)/(2a).
        let beta = super::div(pool.func("sqrt", vec![neg_disc]), two_a, pool);
        // `√(4k²)/2` → `|k|` → `k`: the pair {cos βx, sin βx} spans the same
        // space for β and −β, so a sign-only factor `|u|` of β may be dropped.
        let beta = {
            let _scope = (!ctx.assumptions.predicates().is_empty())
                .then(|| crate::simplify::enter_assumptions(ctx.assumptions, pool));
            strip_abs_factors(
                crate::simplify::real_form::to_real_form_ambient(beta, pool),
                pool,
            )
        };
        let e = exp_of_root(alpha, x, pool);
        let bx = simp(pool.mul(vec![beta, x]), pool);
        let cos = simp(pool.mul(vec![e, pool.func("cos", vec![bx])]), pool);
        let sin = simp(pool.mul(vec![e, pool.func("sin", vec![bx])]), pool);
        return Ok((vec![cos, sin], "constant_coefficient_symbolic_oscillatory"));
    }

    // 3./4. The uniform two-exponential form.  `√D` is written as a `Pow` so
    // that `(√D)² → D` is a power collection the simplifier already performs;
    // spelled `sqrt(D)` it is an opaque `Func` and the residual never closes
    // symbolically.
    let mut root_d = simp(pool.pow(disc, pool.rational(1_i64, 2_i64)), pool);
    if sign == AssumedSign::Positive {
        // `√(4k²)` → `2k` once the sign of `k` is known; value-preserving.
        let _scope = (!ctx.assumptions.predicates().is_empty())
            .then(|| crate::simplify::enter_assumptions(ctx.assumptions, pool));
        root_d = crate::simplify::real_form::to_real_form_ambient(root_d, pool);
    }
    let r_plus = super::div(
        pool.add(vec![pool.mul(vec![neg_one, b]), root_d]),
        two_a,
        pool,
    );
    let r_minus = super::div(
        pool.add(vec![
            pool.mul(vec![neg_one, b]),
            pool.mul(vec![neg_one, root_d]),
        ]),
        two_a,
        pool,
    );
    let basis = vec![exp_of_root(r_plus, x, pool), exp_of_root(r_minus, x, pool)];
    if asserted_nonzero {
        return Ok((basis, "constant_coefficient_symbolic_distinct_roots"));
    }
    ctx.conds.require_nonzero(disc);
    let v = pool.display(x);
    ctx.conds.note(format!(
        "the two exponentials coincide where the characteristic discriminant {d} \
         vanishes; there the general solution is C1·e^({r}·{v}) + C2·{v}·e^({r}·{v}), \
         which this two-parameter family does not span",
        d = pool.display(disc),
        r = pool.display(alpha),
    ));
    Ok((basis, "constant_coefficient_symbolic"))
}

/// Replace every `|u|` that is a direct factor of `beta` by `u`. Only sound
/// where the caller needs `beta` up to sign, as the `{cos βx, sin βx}` basis
/// does.
fn strip_abs_factors(beta: ExprId, pool: &ExprPool) -> ExprId {
    let unabs = |f: ExprId| match pool.get(f) {
        ExprData::Func { name, args } if name == "abs" && args.len() == 1 => args[0],
        _ => f,
    };
    match pool.get(beta) {
        ExprData::Mul(fs) => {
            let new: Vec<ExprId> = fs.iter().map(|&f| unabs(f)).collect();
            if new == fs {
                beta
            } else {
                simp(pool.mul(new), pool)
            }
        }
        _ => unabs(beta),
    }
}

/// `e^{r·x}`, collapsing to `1` when `r` is provably zero.
fn exp_of_root(r: ExprId, x: ExprId, pool: &ExprPool) -> ExprId {
    if matches!(zero_status(pool, r), ZeroStatus::Zero) {
        return pool.integer(1_i32);
    }
    simp(
        pool.func("exp", vec![simp(pool.mul(vec![r, x]), pool)]),
        pool,
    )
}

/// Record `e ≠ 0` unless it is already known — a non-zero literal, or a fact
/// the caller stated.
fn require_nonzero(e: ExprId, ctx: &mut SolveCtx<'_>, pool: &ExprPool) {
    if super::is_settled_nonzero(e, pool) || ctx.asserts_nonzero(e, pool) {
        return;
    }
    ctx.conds.require_nonzero(simp(e, pool));
}

// ---------------------------------------------------------------------------
// Particular solution: exact undetermined coefficients, then variation of
// parameters
// ---------------------------------------------------------------------------

fn particular_solution(
    input: &OdeInput,
    coeffs: &[ExprId],
    rational: Option<&[Rational]>,
    basis: &[ExprId],
    r: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, DsolveError> {
    // Exact undetermined coefficients over ℚ(i) for a forcing term made of
    // `c·xᵏ·e^{ax}·{1, cos bx, sin bx}` with rational `c, a, b`.
    if let Some(p) = rational {
        let r_exp = simp(r, pool);
        if let Some(fams) = exact_roots::forcing_families(r_exp, input.x, pool) {
            if let Some(yp) = exact_roots::particular_exact(p, &fams, input.x, pool) {
                return Ok(simp(yp, pool));
            }
        }
    }
    // Variation of parameters (any order): normalise by the leading
    // coefficient, then quadrature against the fundamental system.
    if basis.len() == input.order() {
        let g = super::div(r, coeffs[input.order()], pool);
        if let Some(yp) = super::variation::variation_of_parameters(basis, g, input.x, pool)? {
            return Ok(yp);
        }
    }
    Err(DsolveError::Unsupported(
        "could not find a particular solution (RHS not handled / integral did not close)"
            .to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Euler–Cauchy: a x² y'' + b x y' + c y = 0.  Substitution y = x^m gives the
// indicial equation a m(m−1) + b m + c = 0.
// ---------------------------------------------------------------------------

fn try_euler_cauchy(
    input: &OdeInput,
    coeffs: &[ExprId],
    r: ExprId,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<Vec<DsolveBranch>>, DsolveError> {
    let x = input.x;
    // Require coeffs[k] = c_k · x^k with rational c_k.
    let mut c: Vec<Rational> = Vec::with_capacity(coeffs.len());
    for (k, &coeff) in coeffs.iter().enumerate() {
        // c_k = coeff · x^{−k} must be constant.  Multiply by a single negative
        // power so the exponents combine (`x^k · x^{−k} → x^0 = 1`); a nested
        // `(x^k)^{−1}` would not flatten.
        let ck = if k == 0 {
            simp(coeff, pool)
        } else {
            let xnegk = pool.pow(x, pool.integer(-(k as i32)));
            simp(pool.mul(vec![coeff, xnegk]), pool)
        };
        if contains(ck, x, pool) {
            return Ok(None);
        }
        // A symbolic (but x-free) coefficient means this is not the rational
        // Euler–Cauchy this route handles.  Decline the *class*, not the
        // equation: reduction of order gets its turn next.
        let Some(v) = exact_roots::literal(ck, pool) else {
            return Ok(None);
        };
        c.push(v);
    }
    let n = c.len() - 1;
    if n == 0 || c[n] == 0 {
        return Ok(None);
    }
    // Indicial polynomial Σ c_k · m(m−1)…(m−k+1).
    let mut ind = vec![Rational::new(); n + 1];
    for (k, ck) in c.iter().enumerate() {
        // falling factorial m^(k) as coefficients, low → high
        let mut ff = vec![Rational::from(1)];
        for j in 0..k {
            let mut next = vec![Rational::new(); ff.len() + 1];
            for (i, v) in ff.iter().enumerate() {
                next[i + 1] += v.clone();
                next[i] -= v.clone() * Rational::from(j as i64);
            }
            ff = next;
        }
        for (i, v) in ff.iter().enumerate() {
            ind[i] += v.clone() * ck.clone();
        }
    }
    let roots = exact_roots::exact_roots(&ind, pool)?;
    let basis = basis_from_roots(&roots, x, true, pool);
    if basis.len() != n {
        return Ok(None);
    }
    compose_from_basis(input, &basis, coeffs[n], r, "euler_cauchy", gen, pool)
}

/// Assemble `y = Σ Cₖ yₖ (+ y_p)` from a fundamental system, verify it, and
/// wrap it as a branch list.
///
/// `leading` is the coefficient of the highest derivative *as it appears in
/// the equation* — variation of parameters needs the equation in monic form,
/// so the forcing term is `g = r / leading`.
///
/// Only the homogeneous part allocates constants; the particular solution
/// contributes none.  (A constant of integration inside `y_p` would be a
/// multiple of some `yₖ`, i.e. already spanned by the `Cₖ` — but allocating a
/// *fresh* symbol for it would leave `dsolve` claiming more free constants
/// than the equation has, which is the classic way this goes wrong.)
fn compose_from_basis(
    input: &OdeInput,
    basis: &[ExprId],
    leading: ExprId,
    r: ExprId,
    method: &'static str,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<Vec<DsolveBranch>>, DsolveError> {
    let mut constants = Vec::with_capacity(basis.len());
    let mut terms = Vec::with_capacity(basis.len());
    for &b in basis {
        let c = gen.fresh(pool);
        constants.push(c);
        terms.push(pool.mul(vec![c, b]));
    }
    let mut y_expr = simp(pool.add(terms), pool);
    if !is_zero(r, pool) {
        let g = super::div(r, leading, pool);
        let Some(yp) = super::variation::variation_of_parameters(basis, g, input.x, pool)? else {
            return Ok(None);
        };
        y_expr = simp(pool.add(vec![y_expr, yp]), pool);
    }
    match residual_is_zero(input, y_expr, &constants, pool) {
        Ok(()) => Ok(Some(vec![DsolveBranch::explicit(
            y_expr, constants, method,
        )])),
        Err(_) => Ok(None),
    }
}
