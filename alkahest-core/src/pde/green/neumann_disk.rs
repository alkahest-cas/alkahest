//! The Neumann problem on a disk: the Neumann function, the compatibility
//! condition, and the solution (unique up to an additive constant).
//!
//! On the disk `|x| < R` the Neumann function (Green's function of the second
//! kind) is
//!
//! `N(x, ξ) = (1/4π)[log|x − ξ|² + log(Q/R²)] − (|x|² + |ξ|²)/(4πR²)`,
//! `Q = |ξ|²|x|² − 2R² x·ξ + R⁴ = |ξ|²|x − ξ*|²`, `ξ* = R²ξ/|ξ|²`,
//!
//! with `Δₓ N = δ(x − ξ) − 1/(πR²)` and `∂N/∂n = 0` on the circle (no
//! Green's function with `ΔG = δ` and `∂G/∂n = 0` exists, since the flux of
//! `δ` through the boundary would have to vanish).  `N` is symmetric, and
//! determined up to an additive function of `ξ`.
//!
//! The Neumann problem `Δu = f` in the disk, `∂u/∂r = g` on `r = R` has a
//! solution if and only if `∫_D f dA = ∮ g ds` (the divergence theorem); it is
//! then unique up to a constant (the difference of two solutions is harmonic
//! with zero normal derivative, so `∫|∇w|² = ∮ w ∂ₙw = 0`), and
//! `u(x) = ū + ∫_D N(x, ξ) f(ξ) dξ − ∮ N(x, ξ) g(ξ) ds_ξ`.
//!
//! [`neumann_disk_solve`] solves it in closed form for a polynomial source
//! and trigonometric-polynomial data `g(θ)`: a polynomial particular solution
//! of `Δv = f`, the Fourier coefficients of `g − ∂ᵣv` by an exact Chebyshev
//! expansion (`cos jθ = T_j(cos θ)`, `sin jθ = sin θ U_{j−1}(cos θ)`), and the
//! harmonic part `Σ (R/j)(r/R)^j (a_j cos jθ + b_j sin jθ)`.  Every step is
//! checked exactly: `Δu = f` by substitution, and `∂ᵣu = g` on the circle as a
//! polynomial identity in `(cos θ, sin θ)` modulo `cos²θ + sin²θ = 1`.

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::ZeroStatus;

use super::super::error::PdeError;
use super::super::sturm_liouville::{replace_node, simp, subs1};
use super::super::verify::VerificationStatus;
use super::{
    check, check_all, combine, dd, div, expand_poly, fresh, fundamental_solution,
    image_green_function, is_symbol, mentions, neg, pi, refuse_refuted, sq, sub, zstatus,
    GreenCheck, GreenFunction, GreenOperator, ImageRegion, WallCondition,
};

fn int(n: i64, pool: &ExprPool) -> ExprId {
    pool.integer(n)
}

fn dot(a: &[ExprId], b: &[ExprId], pool: &ExprPool) -> ExprId {
    pool.add(
        a.iter()
            .zip(b)
            .map(|(&u, &v)| pool.mul(vec![u, v]))
            .collect(),
    )
}

/// The Neumann function of the disk `|x| < radius` (two space variables).
pub(super) fn neumann_function_disk(
    space: &[ExprId],
    source: &[ExprId],
    radius: ExprId,
    pool: &ExprPool,
) -> Result<GreenFunction, PdeError> {
    let r2 = sq(radius, pool);
    let x2 = dot(space, space, pool);
    let xi2 = dot(source, source, pool);
    let xxi = dot(space, source, pool);
    let d2 = pool.add(
        space
            .iter()
            .zip(source)
            .map(|(&u, &v)| sq(sub(u, v, pool), pool))
            .collect(),
    );
    let q = pool.add(vec![
        pool.mul(vec![xi2, x2]),
        neg(pool.mul(vec![int(2, pool), r2, xxi]), pool),
        sq(r2, pool),
    ]);
    let four_pi = pool.mul(vec![int(4, pool), pi(pool)]);
    let quad = neg(
        div(pool.add(vec![x2, xi2]), pool.mul(vec![four_pi, r2]), pool),
        pool,
    );
    let mut checks: Vec<GreenCheck> = Vec::new();
    // The singular part is the table kernel log|x − ξ|/(2π), translated.
    let table = fundamental_solution(GreenOperator::Laplace, space, None, pool)?;
    let mut shift = std::collections::HashMap::new();
    for (&v, &p) in space.iter().zip(source) {
        shift.insert(v, sub(v, p, pool));
    }
    let singular = simp(crate::kernel::subs::subs(table.kernel, &shift, pool), pool);
    let image = div(pool.func("log", vec![div(q, r2, pool)]), four_pi, pool);
    let n_fun = pool.add(vec![singular, image, quad]);
    checks.push(GreenCheck {
        what: "source",
        status: if table.status == VerificationStatus::ExactlyVerified {
            ZeroStatus::ProvedZero
        } else {
            ZeroStatus::Unknown
        },
        method: "fundamental_solution(laplace, 2-D), translated".into(),
        detail: "the singular part of N is the free-space kernel log|x − ξ|/(2π) (unit flux \
                 at ξ, ΔN = δ there); the rest is smooth in the disk (kelvin)"
            .into(),
    });
    // Δₓ N = −1/(πR²) away from ξ: each part.
    let parts = [
        GreenOperator::Laplace.apply(singular, space, None, pool)?,
        GreenOperator::Laplace.apply(pool.func("log", vec![q]), space, None, pool)?,
        pool.add(vec![
            GreenOperator::Laplace.apply(quad, space, None, pool)?,
            div(int(1, pool), pool.mul(vec![pi(pool), r2]), pool),
        ]),
    ];
    checks.push(check_all(
        "pde",
        &parts,
        "substitution (source term, image term and the quadratic separately)",
        "Δₓ N = −1/(πR²) for x ≠ ξ in the disk: ΔN = δ(x − ξ) − 1/(πR²)",
        pool,
    ));
    let kel = sub(
        sub(q, pool.mul(vec![r2, d2]), pool),
        pool.mul(vec![sub(x2, r2, pool), sub(xi2, r2, pool)]),
        pool,
    );
    checks.push(check(
        "kelvin",
        expand_poly(kel, pool),
        "polynomial identity",
        "Q − R²|x − ξ|² = (|x|² − R²)(|ξ|² − R²), so Q > 0 for |x|, |ξ| < R: the image term \
         is smooth in the disk",
        pool,
    ));
    // ∂N/∂n = (x·∇ₓN)/R = 0 on |x| = R.
    let mut grad = Vec::new();
    for &v in space {
        grad.push(pool.mul(vec![v, dd(n_fun, v, pool)?]));
    }
    let normal = simp(pool.add(grad), pool);
    let on_circle = simp(subs1(normal, radius, super::sqrt(x2, pool), pool), pool);
    let mut bc = check(
        "bc",
        on_circle,
        "substitution R = |x|",
        "∂N/∂n = 0 on |x| = R",
        pool,
    );
    if bc.status != ZeroStatus::ProvedZero {
        let num = crate::poly::cancel::cancel(on_circle, [space, source].concat(), pool)
            .unwrap_or(on_circle);
        bc.status = zstatus(expand_poly(simp(num, pool), pool), pool);
    }
    checks.push(bc);
    // Symmetry N(x, ξ) = N(ξ, x).
    let mut swap_syms: Vec<ExprId> = Vec::new();
    for _ in space {
        let mut av = vec![n_fun];
        av.extend(swap_syms.iter().copied());
        swap_syms.push(fresh("__swap", Domain::Real, &av, pool));
    }
    let mut m1 = std::collections::HashMap::new();
    for (&v, &s) in space.iter().zip(&swap_syms) {
        m1.insert(v, s);
    }
    let step1 = crate::kernel::subs::subs(n_fun, &m1, pool);
    let mut m2 = std::collections::HashMap::new();
    for (&v, &p) in source.iter().zip(space) {
        m2.insert(v, p);
    }
    let step2 = crate::kernel::subs::subs(step1, &m2, pool);
    let mut m3 = std::collections::HashMap::new();
    for (&s, &p) in swap_syms.iter().zip(source) {
        m3.insert(s, p);
    }
    let swapped = crate::kernel::subs::subs(step2, &m3, pool);
    let mut sym = check(
        "symmetry",
        sub(n_fun, swapped, pool),
        "substitution",
        "N(x, ξ) = N(ξ, x)",
        pool,
    );
    if sym.status != ZeroStatus::ProvedZero {
        sym.status = zstatus(
            super::expand_func_args(sub(n_fun, swapped, pool), pool),
            pool,
        );
    }
    checks.push(sym);
    // The boundary kernel: −N(x, ξ) for |ξ| = R, where Q/R² = |x − ξ|², so
    // the image term equals the singular one.
    let bkernel = pool.add(vec![
        neg(pool.mul(vec![int(2, pool), singular]), pool),
        div(pool.add(vec![x2, r2]), pool.mul(vec![four_pi, r2]), pool),
    ]);
    let on_b = simp(
        subs1(
            sub(div(q, r2, pool), d2, pool),
            radius,
            super::sqrt(xi2, pool),
            pool,
        ),
        pool,
    );
    let mut bk = check(
        "boundary_kernel",
        on_b,
        "substitution R = |ξ|",
        "on |ξ| = R, Q/R² = |x − ξ|², so the image term equals the singular term and \
         −N(x, ξ) = −(1/π) log|x − ξ| + (|x|² + R²)/(4πR²)",
        pool,
    );
    if bk.status != ZeroStatus::ProvedZero {
        let num = crate::poly::cancel::cancel(on_b, [space, source].concat(), pool).unwrap_or(on_b);
        bk.status = zstatus(expand_poly(simp(num, pool), pool), pool);
    }
    checks.push(bk);
    refuse_refuted("the Neumann function of the disk", &checks)?;
    let status = combine(&checks);
    Ok(GreenFunction {
        operator: "laplace".into(),
        method: "neumann_function".into(),
        space: space.to_vec(),
        source: source.to_vec(),
        time: None,
        value: simp(n_fun, pool),
        series: None,
        branches: None,
        poisson_kernel: Some(simp(bkernel, pool)),
        boundary_vars: source.to_vec(),
        checks,
        status,
        side_conditions: vec![pool.pred_lt(xi2, r2), pool.pred_gt(radius, int(0, pool))],
        notes: vec![
            "Neumann function (Green's function of the second kind): Δₓ N = δ(x − ξ) − 1/(πR²), \
             ∂N/∂n = 0 on |x| = R; a Green's function with ΔG = δ and ∂G/∂n = 0 cannot exist \
             (the flux of δ through the circle is 1). N is unique up to an additive function \
             of ξ; this is the symmetric choice"
                .into(),
            "for Δu = f, ∂u/∂n = g with the compatibility condition ∫_D f dA = ∮ g ds: \
             u(x) = ū + ∫_D N(x, ξ) f(ξ) dξ − ∮_{|ξ|=R} N(x, ξ) g(ξ) ds_ξ (ū the mean of u, an \
             arbitrary constant); `poisson_kernel` is the kernel −N(x, ξ) on |ξ| = R that \
             multiplies g"
                .into(),
        ],
        references: vec![
            "Duffy, Green's Functions with Applications §5.4 (Neumann functions)".into(),
            "Stakgold & Holst, Green's Functions and Boundary Value Problems §8.3 (modified \
             Green's functions, Neumann problem)"
                .into(),
            "Haberman, Applied PDE §9.5.8 (Neumann function, compatibility)".into(),
        ],
    })
}

/// The solution of the Neumann problem on a disk: see [`neumann_disk_solve`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct NeumannDiskSolution {
    /// `u(x, y)`, including the arbitrary constant [`Self::constant`].
    pub value: ExprId,
    /// The arbitrary additive constant `C` (the solution is unique up to it).
    pub constant: ExprId,
    /// The polar form `u(r, θ)` (in [`Self::polar_vars`]).
    pub polar: ExprId,
    /// `(r, θ)` of [`Self::polar`].
    pub polar_vars: (ExprId, ExprId),
    /// `∫_D f dA − ∮ g ds` divided by `2πR`: the mean of `g − ∂ᵣv` over the
    /// circle, proved zero (the compatibility condition).
    pub compatibility: ExprId,
    /// The Fourier coefficients `(j, a_j, b_j)` of the effective boundary
    /// data `g − ∂ᵣv` (`j ≥ 1`).
    pub coefficients: Vec<(i64, ExprId, ExprId)>,
    /// The polynomial particular solution `v` of `Δv = f`.
    pub particular: ExprId,
    /// The Neumann function used for the integral representation.
    pub green: GreenFunction,
    /// The exact checks.
    pub checks: Vec<GreenCheck>,
    /// Exactly verified only when every check is a proof.
    pub status: VerificationStatus,
    /// Conditions (predicates).
    pub side_conditions: Vec<ExprId>,
    /// Hypotheses and the uniqueness statement, in prose.
    pub notes: Vec<String>,
    /// References.
    pub references: Vec<String>,
}

/// Whether `e` is a polynomial in `vars` (coefficients free of them).
fn is_polynomial(e: ExprId, vars: &[ExprId], pool: &ExprPool) -> bool {
    if vars.iter().all(|&v| !mentions(e, v, pool)) {
        return true;
    }
    match pool.get(e) {
        ExprData::Symbol { .. } => true,
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().all(|&x| is_polynomial(x, vars, pool)),
        ExprData::Pow { base, exp } => {
            let k = pool.with(exp, |d| match d {
                ExprData::Integer(i) => i.0.to_i64(),
                _ => None,
            });
            matches!(k, Some(k) if k >= 0) && is_polynomial(base, vars, pool)
        }
        _ => false,
    }
}

/// A polynomial `v` with `Δv = f` for a polynomial `f(x, y)`: for each
/// homogeneous part `f_m`, `v_m = Σ_k a_k |x|^{2k+2} Δ^k f_m` with
/// `a_0 = 1/(4(m+1))`, `a_k = −a_{k−1}/(4(k+1)(m−k+1))`.
fn particular(f: ExprId, x: ExprId, y: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    let f = simp(expand_poly(f, pool), pool);
    if f == int(0, pool) {
        return Ok(f);
    }
    // Homogeneous parts via the scaling x → s x, y → s y.
    let s = fresh("__s", Domain::Real, &[f, x, y], pool);
    let mut m = std::collections::HashMap::new();
    m.insert(x, pool.mul(vec![s, x]));
    m.insert(y, pool.mul(vec![s, y]));
    let fs = simp(
        expand_poly(crate::kernel::subs::subs(f, &m, pool), pool),
        pool,
    );
    let rho2 = pool.add(vec![sq(x, pool), sq(y, pool)]);
    let mut out = Vec::new();
    let mut deriv = fs;
    let mut fact: i64 = 1;
    for deg in 0..=64i64 {
        let part = simp(
            div(subs1(deriv, s, int(0, pool), pool), int(fact, pool), pool),
            pool,
        );
        if part != int(0, pool) {
            let mut a = pool.rational(1, 4 * (deg + 1));
            let mut lap = part;
            let mut k = 0i64;
            while simp(lap, pool) != int(0, pool) && 2 * k <= deg {
                out.push(pool.mul(vec![a, pool.pow(rho2, int(k + 1, pool)), lap]));
                k += 1;
                a = simp(
                    neg(div(a, int(4 * (k + 1) * (deg - k + 1), pool), pool), pool),
                    pool,
                );
                lap = simp(
                    expand_poly(
                        GreenOperator::Laplace.apply(lap, &[x, y], None, pool)?,
                        pool,
                    ),
                    pool,
                );
            }
        }
        deriv = simp(dd(deriv, s, pool)?, pool);
        if deriv == int(0, pool) {
            break;
        }
        if deg == 64 {
            return Err(PdeError::GreenDataUnsupported(
                "the source polynomial has degree above 64".into(),
            ));
        }
        fact = fact.saturating_mul(deg + 1);
    }
    Ok(simp(expand_poly(pool.add(out), pool), pool))
}

/// Chebyshev polynomials `T_j(c)` and `U_j(c)` for `j ≤ jmax`, expanded.
fn chebyshev(c: ExprId, jmax: usize, pool: &ExprPool) -> (Vec<ExprId>, Vec<ExprId>) {
    let two_c = pool.mul(vec![int(2, pool), c]);
    let mut t = vec![int(1, pool), c];
    let mut u = vec![int(1, pool), two_c];
    for j in 2..=jmax.max(1) {
        let tn = simp(
            expand_poly(sub(pool.mul(vec![two_c, t[j - 1]]), t[j - 2], pool), pool),
            pool,
        );
        let un = simp(
            expand_poly(sub(pool.mul(vec![two_c, u[j - 1]]), u[j - 2], pool), pool),
            pool,
        );
        t.push(tn);
        u.push(un);
    }
    (t, u)
}

/// The trigonometric degree of `e` in `θ` (sin / cos of integer multiples).
/// `None` when `θ` enters any other way.
fn trig_degree(e: ExprId, theta: ExprId, pool: &ExprPool) -> Option<i64> {
    if !mentions(e, theta, pool) {
        return Some(0);
    }
    match pool.get(e) {
        ExprData::Add(xs) => {
            let mut m = 0;
            for x in xs {
                m = m.max(trig_degree(x, theta, pool)?);
            }
            Some(m)
        }
        ExprData::Mul(xs) => {
            let mut m = 0;
            for x in xs {
                m += trig_degree(x, theta, pool)?;
            }
            Some(m)
        }
        ExprData::Pow { base, exp } => {
            let k = pool.with(exp, |d| match d {
                ExprData::Integer(i) => i.0.to_i64(),
                _ => None,
            })?;
            if k < 0 {
                return None;
            }
            Some(k * trig_degree(base, theta, pool)?)
        }
        ExprData::Func { name, args } if args.len() == 1 && (name == "sin" || name == "cos") => {
            multiple(args[0], theta, pool).map(|j| j.abs())
        }
        _ => None,
    }
}

/// `j` when `e = j θ` for an integer `j`.
fn multiple(e: ExprId, theta: ExprId, pool: &ExprPool) -> Option<i64> {
    let j = simp(div(e, theta, pool), pool);
    let j = pool.with(j, |d| match d {
        ExprData::Integer(i) => i.0.to_i64(),
        _ => None,
    })?;
    (simp(sub(e, pool.mul(vec![int(j, pool), theta]), pool), pool) == int(0, pool)).then_some(j)
}

/// Rewrite a trigonometric polynomial in `θ` as a polynomial in `c = cos θ`,
/// `s = sin θ`, reduced to degree ≤ 1 in `s` by `s² = 1 − c²`.
fn to_circle(
    e: ExprId,
    theta: ExprId,
    c: ExprId,
    s: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let deg = trig_degree(e, theta, pool).ok_or_else(|| {
        PdeError::GreenDataUnsupported(format!(
            "{} is not a trigonometric polynomial in {} (sums of products of sin(jθ), cos(jθ) \
             with integer j)",
            pool.display(e),
            pool.display(theta)
        ))
    })?;
    let (t, u) = chebyshev(c, deg.max(1) as usize, pool);
    let mut out = e;
    let mut stack = vec![e];
    let mut nodes = Vec::new();
    while let Some(n) = stack.pop() {
        if let ExprData::Func { name, args } = pool.get(n) {
            if args.len() == 1 && (name == "sin" || name == "cos") && mentions(args[0], theta, pool)
            {
                nodes.push((name, args[0], n));
                continue;
            }
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
    for (name, arg, n) in nodes {
        let j = multiple(arg, theta, pool).expect("checked by trig_degree");
        let ja = j.unsigned_abs() as usize;
        let rep = if name == "cos" {
            t[ja]
        } else if ja == 0 {
            int(0, pool)
        } else {
            let v = pool.mul(vec![s, u[ja - 1]]);
            if j < 0 {
                neg(v, pool)
            } else {
                v
            }
        };
        out = replace_node(out, n, rep, pool);
    }
    reduce_s(out, c, s, pool)
}

/// Expand and replace `s^k` by `(1 − c²)^{⌊k/2⌋} s^{k mod 2}`.
fn reduce_s(e: ExprId, c: ExprId, s: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    let one_m = sub(int(1, pool), sq(c, pool), pool);
    let mut cur = simp(expand_poly(e, pool), pool);
    for _ in 0..8 {
        let mut stack = vec![cur];
        let mut nodes = Vec::new();
        while let Some(n) = stack.pop() {
            if let ExprData::Pow { base, exp } = pool.get(n) {
                if base == s {
                    if let Some(k) = pool.with(exp, |d| match d {
                        ExprData::Integer(i) => i.0.to_i64(),
                        _ => None,
                    }) {
                        if k >= 2 {
                            nodes.push((n, k));
                            continue;
                        }
                    }
                }
            }
            super::super::jet::push_children(n, pool, &mut stack);
        }
        if nodes.is_empty() {
            return Ok(cur);
        }
        for (n, k) in nodes {
            let rep = pool.mul(vec![
                pool.pow(one_m, int(k / 2, pool)),
                pool.pow(s, int(k % 2, pool)),
            ]);
            cur = replace_node(cur, n, rep, pool);
        }
        cur = simp(expand_poly(cur, pool), pool);
    }
    Err(PdeError::GreenDataUnsupported(
        "the boundary data did not reduce to a polynomial in cos θ, sin θ".into(),
    ))
}

/// Coefficients of a polynomial `p(c)` of degree ≤ `deg`: `[p_0, …, p_deg]`.
fn coefficients(
    p: ExprId,
    c: ExprId,
    deg: usize,
    pool: &ExprPool,
) -> Result<Vec<ExprId>, PdeError> {
    let mut out = Vec::new();
    let mut d = p;
    let mut fact: i64 = 1;
    for j in 0..=deg {
        out.push(simp(
            div(subs1(d, c, int(0, pool), pool), int(fact, pool), pool),
            pool,
        ));
        d = simp(dd(d, c, pool)?, pool);
        fact = fact.saturating_mul(j as i64 + 1);
    }
    if zstatus(d, pool) != ZeroStatus::ProvedZero {
        return Err(PdeError::GreenDataUnsupported(
            "the boundary data has a higher degree than its trigonometric degree".into(),
        ));
    }
    Ok(out)
}

/// The coefficients of `p(c) = Σ_j a_j T_j(c)` (or `Σ_j b_j U_j(c)`).
fn chebyshev_coefficients(
    p: ExprId,
    c: ExprId,
    basis: &[ExprId],
    pool: &ExprPool,
) -> Result<Vec<ExprId>, PdeError> {
    let deg = basis.len() - 1;
    let mut rest = p;
    let mut out = vec![int(0, pool); deg + 1];
    for j in (0..=deg).rev() {
        let cs = coefficients(rest, c, deg, pool)?;
        let lead_basis = coefficients(basis[j], c, deg, pool)?[j];
        let a = simp(div(cs[j], lead_basis, pool), pool);
        out[j] = a;
        rest = simp(
            expand_poly(sub(rest, pool.mul(vec![a, basis[j]]), pool), pool),
            pool,
        );
    }
    Ok(out)
}

/// Solve `Δu = f` in the disk `x² + y² < R²`, `∂u/∂r = g(θ)` on the circle
/// `x = R cos θ`, `y = R sin θ`, for a polynomial source `f(x, y)` and
/// trigonometric-polynomial data `g(θ)`: a polynomial particular solution, the
/// exact Chebyshev expansion of the boundary data and the harmonic part
/// `Σ (R/j)(r/R)^j (a_j cos jθ + b_j sin jθ)`.
///
/// The compatibility condition `∫_D f dA = ∮ g ds` is decided exactly (it is
/// the vanishing of the mean of `g − ∂ᵣv` over the circle); the solution is
/// returned with an arbitrary additive constant, which is the whole
/// non-uniqueness.  The result is verified by substitution: `Δu = f`, and
/// `∂ᵣu = g` on the circle as a polynomial identity in `cos θ`, `sin θ`.
///
/// # Errors
///
/// * `E-PDE-251` — `x`, `y`, `θ` not distinct symbols;
/// * `E-PDE-257` — the compatibility condition fails: no solution exists;
/// * `E-PDE-258` — the data are outside the closed-form class (a source that
///   is not a polynomial, data that are not a trigonometric polynomial), or
///   the compatibility condition cannot be decided (symbolic coefficients);
/// * `E-PDE-255` — a check refuted the construction (never returned).
pub fn neumann_disk_solve(
    x: ExprId,
    y: ExprId,
    radius: ExprId,
    f: ExprId,
    g: ExprId,
    theta: ExprId,
    pool: &ExprPool,
) -> Result<NeumannDiskSolution, PdeError> {
    for v in [x, y, theta] {
        if !is_symbol(v, pool) {
            return Err(PdeError::GreenInvalidInput(format!(
                "{} is not a symbol",
                pool.display(v)
            )));
        }
    }
    if x == y || x == theta || y == theta {
        return Err(PdeError::GreenInvalidInput(
            "x, y and θ must be distinct symbols".into(),
        ));
    }
    if mentions(f, theta, pool) || mentions(g, x, pool) || mentions(g, y, pool) {
        return Err(PdeError::GreenInvalidInput(
            "the source is a function of (x, y) and the boundary data a function of θ".into(),
        ));
    }
    if [x, y, theta].iter().any(|&v| mentions(radius, v, pool)) {
        return Err(PdeError::GreenInvalidInput(
            "the radius must not depend on x, y or θ".into(),
        ));
    }
    if !is_polynomial(f, &[x, y], pool) {
        return Err(PdeError::GreenDataUnsupported(format!(
            "the source {} is not a polynomial in {}, {}; use the Neumann function \
             (image_green_function on the disk with a Neumann wall) for the integral \
             representation",
            pool.display(f),
            pool.display(x),
            pool.display(y)
        )));
    }
    let avoid = [x, y, theta, radius, f, g];
    let c = fresh("__c", Domain::Real, &avoid, pool);
    let s = fresh("__s", Domain::Real, &avoid, pool);
    let v = particular(f, x, y, pool)?;
    let radial_d = |w: ExprId| -> Result<ExprId, PdeError> {
        Ok(div(
            pool.add(vec![
                pool.mul(vec![x, dd(w, x, pool)?]),
                pool.mul(vec![y, dd(w, y, pool)?]),
            ]),
            radius,
            pool,
        ))
    };
    let on_circle_cs = |w: ExprId| {
        let mut m = std::collections::HashMap::new();
        m.insert(x, pool.mul(vec![radius, c]));
        m.insert(y, pool.mul(vec![radius, s]));
        crate::kernel::subs::subs(w, &m, pool)
    };
    // h = g − ∂ᵣv on the circle, as P(c) + s Q(c).
    let g_cs = to_circle(g, theta, c, s, pool)?;
    let h = reduce_s(sub(g_cs, on_circle_cs(radial_d(v)?), pool), c, s, pool)?;
    let p_part = simp(subs1(h, s, int(0, pool), pool), pool);
    let q_part = simp(dd(h, s, pool)?, pool);
    let deg_g = trig_degree(g, theta, pool).unwrap_or(0);
    let deg_f = {
        // the degree of ∂ᵣv on the circle is at most deg f + 1
        let mut d = 0usize;
        let mut cur = simp(expand_poly(f, pool), pool);
        while zstatus(cur, pool) != ZeroStatus::ProvedZero && d < 70 {
            cur = simp(dd(cur, x, pool)?, pool);
            d += 1;
        }
        let mut e = 0usize;
        let mut cur = simp(expand_poly(f, pool), pool);
        while zstatus(cur, pool) != ZeroStatus::ProvedZero && e < 70 {
            cur = simp(dd(cur, y, pool)?, pool);
            e += 1;
        }
        (d + e) as i64 + 2
    };
    let deg = (deg_g.max(deg_f)).max(1) as usize;
    let (t_basis, u_basis) = chebyshev(c, deg, pool);
    let a = chebyshev_coefficients(p_part, c, &t_basis[..=deg], pool)?;
    let b = if deg >= 1 {
        chebyshev_coefficients(q_part, c, &u_basis[..deg], pool)?
    } else {
        Vec::new()
    };
    // Compatibility: a₀ = 0.
    let compat = simp(a[0], pool);
    match zstatus(compat, pool) {
        ZeroStatus::ProvedZero => {}
        ZeroStatus::ProvedNonzero
            if !has_parameters(compat, pool)
                || matches!(
                    super::super::sturm_liouville::sign(compat, pool),
                    super::super::sturm_liouville::Sgn::Pos
                        | super::super::sturm_liouville::Sgn::Neg
                ) =>
        {
            return Err(PdeError::GreenCompatibility(format!(
                "∫_D f dA − ∮ g ds = −2πR · {} ≠ 0: the Neumann problem has no solution",
                pool.display(compat)
            )))
        }
        _ => {
            return Err(PdeError::GreenDataUnsupported(format!(
                "the compatibility condition ∫_D f dA = ∮ g ds reads {} = 0, which depends on \
                 the parameters; substitute values that make it hold",
                pool.display(compat)
            )))
        }
    }
    // Harmonic part: Σ_j (R^{1−j}/j) (a_j Re z^j + b_j Im z^j).
    let mut harm = Vec::new();
    let mut coeffs = Vec::new();
    for j in 1..=deg {
        let aj = a[j];
        let bj = if j - 1 < b.len() {
            b[j - 1]
        } else {
            int(0, pool)
        };
        if simp(aj, pool) == int(0, pool) && simp(bj, pool) == int(0, pool) {
            continue;
        }
        coeffs.push((j as i64, aj, bj));
        let (re, im) = re_im_power(x, y, j, pool);
        harm.push(pool.mul(vec![
            div(
                pool.pow(radius, int(1 - j as i64, pool)),
                int(j as i64, pool),
                pool,
            ),
            pool.add(vec![pool.mul(vec![aj, re]), pool.mul(vec![bj, im])]),
        ]));
    }
    let constant = fresh("C", Domain::Real, &avoid, pool);
    let u = simp(
        pool.add(vec![
            constant,
            v,
            if harm.is_empty() {
                int(0, pool)
            } else {
                pool.add(harm)
            },
        ]),
        pool,
    );
    // ---------------------------------------------------------------- checks
    let mut checks: Vec<GreenCheck> = Vec::new();
    checks.push(check(
        "compatibility",
        compat,
        "exact Chebyshev expansion",
        "the mean of g − ∂ᵣv over the circle vanishes: ∫_D f dA = ∮ g ds",
        pool,
    ));
    let pde_res = simp(
        expand_poly(
            sub(
                GreenOperator::Laplace.apply(u, &[x, y], None, pool)?,
                f,
                pool,
            ),
            pool,
        ),
        pool,
    );
    checks.push(check(
        "pde",
        pde_res,
        "substitution",
        "Δu = f in the disk",
        pool,
    ));
    let bc_res = reduce_s(sub(on_circle_cs(radial_d(u)?), g_cs, pool), c, s, pool)?;
    let mut bc = check(
        "bc",
        bc_res,
        "polynomial identity in cos θ, sin θ modulo cos²θ + sin²θ = 1",
        "∂u/∂r = g(θ) on r = R",
        pool,
    );
    if bc.status != ZeroStatus::ProvedZero {
        bc.status = zstatus(expand_poly(bc_res, pool), pool);
    }
    checks.push(bc);
    let gc = check(
        "data",
        sub(on_circle_cs(g), g, pool),
        "the data do not involve x, y",
        "g is a function of θ alone",
        pool,
    );
    checks.push(gc);
    // The Neumann function, for the representation.
    let xi = fresh("xi", Domain::Real, &avoid, pool);
    let eta = fresh("eta", Domain::Real, &[xi, x, y, theta, radius], pool);
    let green = image_green_function(
        GreenOperator::Laplace,
        &[x, y],
        None,
        &[xi, eta],
        ImageRegion::Disk { radius },
        &[WallCondition::Neumann],
        pool,
    )?;
    refuse_refuted("the Neumann-disk solution", &checks)?;
    let status = combine(&checks);
    let r = fresh("r", Domain::Positive, &avoid, pool);
    let mut pm = std::collections::HashMap::new();
    pm.insert(x, pool.mul(vec![r, pool.func("cos", vec![theta])]));
    pm.insert(y, pool.mul(vec![r, pool.func("sin", vec![theta])]));
    let polar = simp(crate::kernel::subs::subs(u, &pm, pool), pool);
    Ok(NeumannDiskSolution {
        value: u,
        constant,
        polar,
        polar_vars: (r, theta),
        compatibility: compat,
        coefficients: coeffs,
        particular: v,
        green,
        checks,
        status,
        side_conditions: vec![pool.pred_gt(radius, int(0, pool))],
        notes: vec![
            "uniqueness: two solutions differ by w with Δw = 0 and ∂w/∂n = 0, and Green's first \
             identity gives ∫_D |∇w|² = ∮ w ∂w/∂n ds = 0, so w is constant; the constant C is \
             the whole non-uniqueness (fix it by prescribing the mean of u)"
                .into(),
            "existence requires the compatibility condition ∫_D f dA = ∮ g ds (the divergence \
             theorem applied to ∇u); it was decided exactly"
                .into(),
            "the same solution is u = ū + ∫_D N f − ∮ N g ds with the Neumann function `green`"
                .into(),
        ],
        references: vec![
            "Strauss, PDE §6.4 (Neumann problem in a disk) and §7.1 (compatibility)".into(),
            "Haberman, Applied PDE §2.5 (Laplace in a circular disk) and §9.5.8".into(),
            "Evans, PDE §2.2 (uniqueness by energy methods)".into(),
        ],
    })
}

/// `Re (x + iy)^j`, `Im (x + iy)^j`, expanded.
fn re_im_power(x: ExprId, y: ExprId, j: usize, pool: &ExprPool) -> (ExprId, ExprId) {
    let mut re = Vec::new();
    let mut im = Vec::new();
    let mut binom = rug::Integer::from(1);
    for m in 0..=j {
        if m > 0 {
            binom = binom * rug::Integer::from(j - m + 1) / rug::Integer::from(m);
        }
        let mono = pool.mul(vec![
            pool.integer(binom.clone()),
            pool.pow(x, int((j - m) as i64, pool)),
            pool.pow(y, int(m as i64, pool)),
        ]);
        match m % 4 {
            0 => re.push(mono),
            1 => im.push(mono),
            2 => re.push(neg(mono, pool)),
            _ => im.push(neg(mono, pool)),
        }
    }
    let tot = |v: Vec<ExprId>| {
        if v.is_empty() {
            int(0, pool)
        } else {
            simp(pool.add(v), pool)
        }
    };
    (tot(re), tot(im))
}

/// Whether `e` has a free symbol (other than `π`).
fn has_parameters(e: ExprId, pool: &ExprPool) -> bool {
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if let ExprData::Symbol { name, .. } = pool.get(n) {
            if name != "pi" {
                return true;
            }
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
    false
}
