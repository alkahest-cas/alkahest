//! Linear constant-coefficient and Euler–Cauchy ODE classes for
//! [`super::dsolve`], plus higher-order constant-coefficient.
//!
//! Coefficients of `y^(k)` are extracted from the equation.  When all are
//! constant the characteristic polynomial is built and factored over ℚ
//! (rational roots + quadratic factors); irreducible factors of degree ≥ 3 are
//! declined.  Euler–Cauchy `a·x²y'' + b·x·y' + c·y = 0` is detected by the
//! `xᵏ` coefficient pattern.

use super::{
    contains, ddx, integrate_or_decline, is_zero, residual_is_zero, simp, sub, ConstGen,
    DsolveError, DsolveResult, DsolveSolution, OdeInput,
};
use crate::kernel::eval_const::try_expr_f64;
use crate::kernel::{ExprData, ExprId, ExprPool};

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
    pool: &ExprPool,
) -> Result<DsolveResult, DsolveError> {
    let (coeffs, r) = extract_linear(input, pool)?;
    let x = input.x;

    if all_constant(&coeffs, x, pool) {
        return solve_const_coeff(input, &coeffs, r, gen, pool);
    }
    // Euler–Cauchy: coeffs[k] = c_k · x^k with c_k constant.
    if let Some(euler) = try_euler_cauchy(input, &coeffs, r, gen, pool)? {
        return Ok(euler);
    }
    Err(DsolveError::Unsupported(
        "second-order equation is neither constant-coefficient nor Euler–Cauchy".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Constant-coefficient (any order ≥ 1): homogeneous via characteristic poly
// ---------------------------------------------------------------------------

pub(crate) fn solve_higher_order(
    input: &OdeInput,
    _n: usize,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<DsolveResult, DsolveError> {
    let (coeffs, r) = extract_linear(input, pool)?;
    let x = input.x;
    if !all_constant(&coeffs, x, pool) {
        return Err(DsolveError::Unsupported(
            "higher-order non-constant-coefficient equations are not supported".to_string(),
        ));
    }
    solve_const_coeff(input, &coeffs, r, gen, pool)
}

fn solve_const_coeff(
    input: &OdeInput,
    coeffs: &[ExprId],
    r: ExprId,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<DsolveResult, DsolveError> {
    let x = input.x;
    // Numeric coefficients for the characteristic polynomial Σ aₖ λᵏ.
    let mut a: Vec<f64> = Vec::with_capacity(coeffs.len());
    for &c in coeffs {
        let v = try_expr_f64(c, pool).ok_or_else(|| {
            DsolveError::Unsupported("non-numeric constant coefficient".to_string())
        })?;
        a.push(v);
    }
    // Drop leading zeros (highest derivative might be zero).
    while a.len() > 1 && a.last() == Some(&0.0) {
        a.pop();
    }
    let roots = char_roots(&a)?;

    // Build homogeneous basis functions and the general homogeneous solution.
    let mut basis: Vec<ExprId> = Vec::new();
    for root in &roots {
        basis.extend(root.basis_functions(x, pool));
    }
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
        let yp = particular_solution(input, coeffs, &basis, r, pool)?;
        simp(pool.add(vec![y_homog, yp]), pool)
    };

    match residual_is_zero(input, y_general, &constants, pool) {
        Ok(()) => Ok(DsolveResult {
            solutions: vec![DsolveSolution {
                y_of_x: y_general,
                constants,
                method: "constant_coefficient",
            }],
        }),
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// Characteristic roots over ℚ: rational roots + quadratic factors
// ---------------------------------------------------------------------------

/// A root of the characteristic polynomial together with its multiplicity.
enum CharRoot {
    /// Real root `r` (rational) with multiplicity `m`.
    Real { r: f64, m: usize },
    /// Complex conjugate pair `α ± βi` (β > 0) with multiplicity `m`.
    Complex { alpha: f64, beta: f64, m: usize },
}

impl CharRoot {
    /// e^{r x}, x·e^{r x}, … for real; e^{α x}·cos(β x)·xʲ, e^{α x}·sin(β x)·xʲ for complex.
    fn basis_functions(&self, x: ExprId, pool: &ExprPool) -> Vec<ExprId> {
        match *self {
            CharRoot::Real { r, m } => (0..m)
                .map(|j| {
                    let exp_part = exp_rx(r, x, pool);
                    if j == 0 {
                        exp_part
                    } else {
                        let xj = pool.pow(x, pool.integer(j as i32));
                        simp(pool.mul(vec![xj, exp_part]), pool)
                    }
                })
                .collect(),
            CharRoot::Complex { alpha, beta, m } => {
                let mut out = Vec::new();
                for j in 0..m {
                    let exp_part = exp_rx(alpha, x, pool);
                    let bx = mul_const(beta, x, pool);
                    let cos = pool.func("cos", vec![bx]);
                    let sin = pool.func("sin", vec![bx]);
                    for trig in [cos, sin] {
                        let base = simp(pool.mul(vec![exp_part, trig]), pool);
                        let f = if j == 0 {
                            base
                        } else {
                            let xj = pool.pow(x, pool.integer(j as i32));
                            simp(pool.mul(vec![xj, base]), pool)
                        };
                        out.push(f);
                    }
                }
                out
            }
        }
    }
}

fn exp_rx(r: f64, x: ExprId, pool: &ExprPool) -> ExprId {
    if r == 0.0 {
        return pool.integer(1_i32);
    }
    let rx = mul_const(r, x, pool);
    simp(pool.func("exp", vec![rx]), pool)
}

/// `c · x` with `c` rendered as an exact rational when possible.
fn mul_const(c: f64, x: ExprId, pool: &ExprPool) -> ExprId {
    let cexpr = f64_to_expr(c, pool);
    simp(pool.mul(vec![cexpr, x]), pool)
}

/// Convert an `f64` that is a small rational to an exact `ExprId`.
fn f64_to_expr(v: f64, pool: &ExprPool) -> ExprId {
    if v == v.round() {
        return pool.integer(v as i64);
    }
    // Try small denominators.
    for den in 2..=12_i64 {
        let num = v * den as f64;
        if (num - num.round()).abs() < 1e-9 {
            return pool.rational(num.round() as i64, den);
        }
    }
    pool.float(v, 53)
}

/// Compute characteristic roots from coefficients `a[0..=deg]` (a[k] is the
/// coefficient of λᵏ).  Factors out rational roots and quadratic factors;
/// declines on an irreducible factor of degree ≥ 3.
fn char_roots(a: &[f64]) -> Result<Vec<CharRoot>, DsolveError> {
    // Work with the polynomial as a coefficient vector, repeatedly dividing out
    // found roots/factors.  Operate numerically but recover rational roots.
    let mut poly: Vec<f64> = a.to_vec();
    normalize(&mut poly);
    let mut roots: Vec<CharRoot> = Vec::new();

    while poly.len() > 1 {
        let deg = poly.len() - 1;
        if deg == 1 {
            // a0 + a1 λ = 0 → λ = −a0/a1
            let r = -poly[0] / poly[1];
            add_real_root(&mut roots, r);
            break;
        }
        if deg == 2 {
            add_quadratic(&mut roots, poly[2], poly[1], poly[0]);
            break;
        }
        // deg ≥ 3: try to peel off a rational root.
        if let Some(r) = find_rational_root(&poly) {
            add_real_root(&mut roots, r);
            poly = deflate_real(&poly, r);
            normalize(&mut poly);
            continue;
        }
        // try to peel off a rational quadratic factor λ² + p λ + q with rational p,q
        if let Some((p, q)) = find_quadratic_factor(&poly) {
            add_quadratic(&mut roots, 1.0, p, q);
            poly = deflate_quadratic(&poly, p, q);
            normalize(&mut poly);
            continue;
        }
        return Err(DsolveError::Unsupported(
            "characteristic polynomial has an irreducible factor of degree ≥ 3".to_string(),
        ));
    }
    Ok(roots)
}

fn normalize(poly: &mut Vec<f64>) {
    while poly.len() > 1 && poly.last().map(|v| v.abs() < 1e-12).unwrap_or(false) {
        poly.pop();
    }
}

fn add_real_root(roots: &mut Vec<CharRoot>, r: f64) {
    for cr in roots.iter_mut() {
        if let CharRoot::Real { r: rr, m } = cr {
            if (*rr - r).abs() < 1e-7 {
                *m += 1;
                return;
            }
        }
    }
    roots.push(CharRoot::Real { r, m: 1 });
}

fn add_complex_pair(roots: &mut Vec<CharRoot>, alpha: f64, beta: f64) {
    let beta = beta.abs();
    for cr in roots.iter_mut() {
        if let CharRoot::Complex {
            alpha: a,
            beta: b,
            m,
        } = cr
        {
            if (*a - alpha).abs() < 1e-7 && (*b - beta).abs() < 1e-7 {
                *m += 1;
                return;
            }
        }
    }
    roots.push(CharRoot::Complex { alpha, beta, m: 1 });
}

/// Solve `a λ² + b λ + c = 0` and record real/complex roots.
fn add_quadratic(roots: &mut Vec<CharRoot>, a: f64, b: f64, c: f64) {
    let disc = b * b - 4.0 * a * c;
    if disc.abs() < 1e-10 {
        let r = -b / (2.0 * a);
        add_real_root(roots, r);
        add_real_root(roots, r); // double root
    } else if disc > 0.0 {
        let s = disc.sqrt();
        add_real_root(roots, (-b + s) / (2.0 * a));
        add_real_root(roots, (-b - s) / (2.0 * a));
    } else {
        let alpha = -b / (2.0 * a);
        let beta = (-disc).sqrt() / (2.0 * a);
        add_complex_pair(roots, alpha, beta);
    }
}

fn poly_eval(poly: &[f64], x: f64) -> f64 {
    poly.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Try integer candidate roots in a small range (characteristic roots of
/// textbook ODEs are small integers/rationals).
fn find_rational_root(poly: &[f64]) -> Option<f64> {
    for num in -12..=12i64 {
        for den in 1..=6i64 {
            let r = num as f64 / den as f64;
            if poly_eval(poly, r).abs() < 1e-7 {
                return Some(r);
            }
        }
    }
    None
}

fn deflate_real(poly: &[f64], r: f64) -> Vec<f64> {
    // synthetic division by (λ − r); poly is low→high.
    let n = poly.len();
    let mut out = vec![0.0; n - 1];
    let mut carry = 0.0;
    for i in (0..n).rev() {
        if i == 0 {
            break;
        }
        let coeff = poly[i] + carry;
        out[i - 1] = coeff;
        carry = coeff * r;
    }
    out
}

/// Search for a monic rational quadratic factor `λ² + p λ + q`.
fn find_quadratic_factor(poly: &[f64]) -> Option<(f64, f64)> {
    for pn in -12..=12i64 {
        for qn in -12..=12i64 {
            let (p, q) = (pn as f64, qn as f64);
            if divides_quadratic(poly, p, q) {
                return Some((p, q));
            }
        }
    }
    None
}

fn divides_quadratic(poly: &[f64], p: f64, q: f64) -> bool {
    let (quot, rem) = quad_divmod(poly, p, q);
    let _ = quot;
    rem.iter().all(|v| v.abs() < 1e-6)
}

/// Divide `poly` by `λ² + p λ + q`; return `(quotient, remainder[0..2])`.
fn quad_divmod(poly: &[f64], p: f64, q: f64) -> (Vec<f64>, Vec<f64>) {
    // poly low→high; convert to high→low for long division.
    let mut hi: Vec<f64> = poly.iter().rev().cloned().collect();
    let n = hi.len();
    if n < 3 {
        return (vec![0.0], hi);
    }
    let mut quot = vec![0.0; n - 2];
    for i in 0..(n - 2) {
        let c = hi[i];
        quot[i] = c;
        hi[i + 1] -= c * p;
        hi[i + 2] -= c * q;
    }
    let rem = vec![hi[n - 2], hi[n - 1]];
    // quotient is high→low; return low→high
    quot.reverse();
    (quot, rem)
}

fn deflate_quadratic(poly: &[f64], p: f64, q: f64) -> Vec<f64> {
    quad_divmod(poly, p, q).0
}

// ---------------------------------------------------------------------------
// Particular solution: undetermined coefficients, then variation of parameters
// ---------------------------------------------------------------------------

fn particular_solution(
    input: &OdeInput,
    coeffs: &[ExprId],
    basis: &[ExprId],
    r: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, DsolveError> {
    // Try undetermined coefficients first (cheap, exact).
    if let Some(yp) = undetermined_coefficients(input, coeffs, basis, r, pool)? {
        return Ok(yp);
    }
    // Variation of parameters (uses integrate; second order only).
    if input.order() == 2 && basis.len() == 2 {
        if let Some(yp) = variation_of_parameters(input, coeffs, basis, r, pool)? {
            return Ok(yp);
        }
    }
    Err(DsolveError::Unsupported(
        "could not find a particular solution (RHS not handled / integral did not close)"
            .to_string(),
    ))
}

/// Undetermined coefficients for RHS of the form polynomial × exp × {1,cos,sin}.
/// Builds an ansatz with unknown coefficients, substitutes, and solves the
/// resulting linear system by numeric sampling + least-squares-free Gaussian
/// elimination over a small monomial basis.
fn undetermined_coefficients(
    input: &OdeInput,
    coeffs: &[ExprId],
    basis: &[ExprId],
    r: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, DsolveError> {
    let x = input.x;
    // Build the ansatz term list (each term is a basis monomial of the RHS form).
    let Some(ansatz_terms) = ansatz_terms_for(r, x, basis, pool) else {
        return Ok(None);
    };
    if ansatz_terms.is_empty() {
        return Ok(None);
    }
    let k = ansatz_terms.len();

    // L[term_j] evaluated, residual must match r.  Build linear operator L =
    // Σ coeffs[d] · d^d/dx^d.  Sample at k distinct points to solve for the
    // unknown coefficients A_j in  Σ A_j L[term_j] = r.
    let mut l_terms: Vec<ExprId> = Vec::with_capacity(k);
    for &t in &ansatz_terms {
        l_terms.push(apply_operator(coeffs, t, x, pool)?);
    }

    // Set up linear system M · A = b at sample points.
    let samples: Vec<f64> = (0..k).map(|i| 0.37 + 0.53 * i as f64).collect();
    let mut mat = vec![vec![0.0; k]; k];
    let mut rhs_vec = vec![0.0; k];
    for (i, &xv) in samples.iter().enumerate() {
        for (j, &lt) in l_terms.iter().enumerate() {
            mat[i][j] = eval_at(lt, x, xv, pool)
                .ok_or_else(|| DsolveError::Unsupported("ansatz evaluation failed".to_string()))?;
        }
        rhs_vec[i] = eval_at(r, x, xv, pool)
            .ok_or_else(|| DsolveError::Unsupported("rhs evaluation failed".to_string()))?;
    }
    let Some(sol) = solve_linear(&mut mat, &mut rhs_vec) else {
        return Ok(None);
    };
    // Build yp = Σ A_j term_j with rationalised coefficients.
    let mut terms = Vec::new();
    for (j, &t) in ansatz_terms.iter().enumerate() {
        let a = f64_to_expr(sol[j], pool);
        terms.push(pool.mul(vec![a, t]));
    }
    let yp = simp(pool.add(terms), pool);
    Ok(Some(yp))
}

/// Determine the ansatz monomials for an RHS of the supported form.  Handles:
/// polynomial (degree d), poly·exp(a x), poly·{cos,sin}(b x), poly·exp·trig.
/// Multiplies by `x^s` to handle resonance with the homogeneous basis.
fn ansatz_terms_for(
    r: ExprId,
    x: ExprId,
    basis: &[ExprId],
    pool: &ExprPool,
) -> Option<Vec<ExprId>> {
    // Identify exponential rate a, trig rate b, and polynomial degree d.
    let (poly_deg, exp_rate, trig_rate) = classify_rhs(r, x, pool)?;
    // Base modulating factor m(x) = exp(a x) · {1 | cos(bx),sin(bx)}.
    let mut mods: Vec<ExprId> = Vec::new();
    let exp_factor = if exp_rate != 0.0 {
        Some(simp(
            pool.func("exp", vec![mul_const(exp_rate, x, pool)]),
            pool,
        ))
    } else {
        None
    };
    if let Some(b) = trig_rate {
        let bx = mul_const(b, x, pool);
        mods.push(pool.func("cos", vec![bx]));
        mods.push(pool.func("sin", vec![bx]));
    } else {
        mods.push(pool.integer(1_i32));
    }

    // Resonance shift s: how many basis functions match this exp·trig form.
    let s = resonance_shift(exp_rate, trig_rate, basis, x, pool);

    let mut terms = Vec::new();
    for j in 0..=poly_deg {
        let power = j + s;
        let xpow = if power == 0 {
            pool.integer(1_i32)
        } else {
            pool.pow(x, pool.integer(power as i32))
        };
        for &m in &mods {
            let mut fac = vec![xpow, m];
            if let Some(e) = exp_factor {
                fac.push(e);
            }
            terms.push(simp(pool.mul(fac), pool));
        }
    }
    Some(terms)
}

/// Classify the RHS into `(polynomial_degree, exp_rate, Option<trig_rate>)`.
/// Returns None if it is not of the supported product form.
fn classify_rhs(r: ExprId, x: ExprId, pool: &ExprPool) -> Option<(usize, f64, Option<f64>)> {
    // Split off additive structure: for simplicity require a single product
    // term family (the common textbook case).  We detect by scanning factors.
    let factors: Vec<ExprId> = match pool.get(r) {
        ExprData::Mul(args) => args,
        ExprData::Add(_) => {
            // allow pure polynomial sums
            if is_polynomial_in(r, x, pool) {
                return Some((poly_degree(r, x, pool)?, 0.0, None));
            }
            vec![r]
        }
        _ => vec![r],
    };
    let mut exp_rate = 0.0;
    let mut trig_rate: Option<f64> = None;
    let mut poly_factors: Vec<ExprId> = Vec::new();
    for f in factors {
        match pool.get(f) {
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                exp_rate += linear_rate(args[0], x, pool)?;
            }
            ExprData::Func { name, args }
                if (name == "cos" || name == "sin") && args.len() == 1 =>
            {
                let b = linear_rate(args[0], x, pool)?;
                trig_rate = Some(b.abs());
            }
            _ => {
                if contains(f, x, pool) && !is_polynomial_in(f, x, pool) {
                    return None;
                }
                poly_factors.push(f);
            }
        }
    }
    let poly = if poly_factors.is_empty() {
        pool.integer(1_i32)
    } else {
        simp(pool.mul(poly_factors), pool)
    };
    if !is_polynomial_in(poly, x, pool) {
        return None;
    }
    let deg = poly_degree(poly, x, pool)?;
    Some((deg, exp_rate, trig_rate))
}

/// `arg = c · x` → return `c`; only linear-through-origin args are accepted.
fn linear_rate(arg: ExprId, x: ExprId, pool: &ExprPool) -> Option<f64> {
    let d = ddx(arg, x, pool).ok()?;
    if contains(d, x, pool) {
        return None;
    }
    // require arg(0) = 0 (no constant phase): arg − d·x should be 0
    let dx = simp(pool.mul(vec![d, x]), pool);
    let cst = sub(arg, dx, pool);
    if !is_zero(cst, pool) {
        return None;
    }
    try_expr_f64(d, pool)
}

fn is_polynomial_in(expr: ExprId, x: ExprId, pool: &ExprPool) -> bool {
    poly_degree(expr, x, pool).is_some()
}

/// Degree of `expr` as a polynomial in `x`, or None if not polynomial.
fn poly_degree(expr: ExprId, x: ExprId, pool: &ExprPool) -> Option<usize> {
    match pool.get(expr) {
        _ if !contains(expr, x, pool) => Some(0),
        ExprData::Symbol { .. } => Some(1), // == x (since it contains x)
        ExprData::Add(args) | ExprData::Mul(args) => {
            let op_is_mul = matches!(pool.get(expr), ExprData::Mul(_));
            let mut acc = 0usize;
            for a in args {
                let d = poly_degree(a, x, pool)?;
                if op_is_mul {
                    acc += d;
                } else {
                    acc = acc.max(d);
                }
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            if base == x {
                if let ExprData::Integer(k) = pool.get(exp) {
                    let k = k.0.to_i64()?;
                    if k >= 0 {
                        return Some(k as usize);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// How many extra powers of `x` to multiply the ansatz by, to avoid clashing
/// with the homogeneous basis (resonance).  Counts basis functions matching the
/// same exp·trig family.
fn resonance_shift(
    exp_rate: f64,
    trig_rate: Option<f64>,
    basis: &[ExprId],
    x: ExprId,
    pool: &ExprPool,
) -> usize {
    let mut count = 0;
    for &b in basis {
        if basis_matches_family(b, exp_rate, trig_rate, x, pool) {
            count += 1;
        }
    }
    // For real (no trig): each matching basis fn => shift by that many.  For
    // complex pairs the basis contributes cos & sin; we want the multiplicity,
    // so divide by the family width.
    if trig_rate.is_some() {
        count / 2
    } else {
        count
    }
}

fn basis_matches_family(
    b: ExprId,
    exp_rate: f64,
    trig_rate: Option<f64>,
    x: ExprId,
    pool: &ExprPool,
) -> bool {
    // Evaluate b's exp/trig signature by inspecting its factors.
    let (be, bt) = basis_signature(b, x, pool);
    (be - exp_rate).abs() < 1e-7
        && match (bt, trig_rate) {
            (None, None) => true,
            (Some(a), Some(c)) => (a - c).abs() < 1e-7,
            _ => false,
        }
}

fn basis_signature(b: ExprId, x: ExprId, pool: &ExprPool) -> (f64, Option<f64>) {
    let factors: Vec<ExprId> = match pool.get(b) {
        ExprData::Mul(args) => args,
        _ => vec![b],
    };
    let mut e = 0.0;
    let mut t = None;
    for f in factors {
        match pool.get(f) {
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                if let Some(rate) = linear_rate(args[0], x, pool) {
                    e += rate;
                }
            }
            ExprData::Func { name, args }
                if (name == "cos" || name == "sin") && args.len() == 1 =>
            {
                if let Some(rate) = linear_rate(args[0], x, pool) {
                    t = Some(rate.abs());
                }
            }
            _ => {}
        }
    }
    (e, t)
}

/// Apply the linear operator `L = Σ coeffs[d]·d^d/dx^d` to `expr`.
fn apply_operator(
    coeffs: &[ExprId],
    expr: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, DsolveError> {
    let mut acc = Vec::new();
    let mut cur = expr;
    for (d, &c) in coeffs.iter().enumerate() {
        if d > 0 {
            cur = ddx(cur, x, pool)?;
        }
        acc.push(pool.mul(vec![c, cur]));
    }
    Ok(simp(pool.add(acc), pool))
}

fn eval_at(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> Option<f64> {
    use std::collections::HashMap;
    let mut env = HashMap::new();
    env.insert(x, xv);
    super::verify::eval(expr, &env, pool)
}

/// Gaussian elimination with partial pivoting.  Returns None on singularity.
#[allow(clippy::needless_range_loop)]
fn solve_linear(mat: &mut [Vec<f64>], rhs: &mut [f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    for col in 0..n {
        // pivot
        let mut piv = col;
        for r in (col + 1)..n {
            if mat[r][col].abs() > mat[piv][col].abs() {
                piv = r;
            }
        }
        if mat[piv][col].abs() < 1e-12 {
            return None;
        }
        mat.swap(col, piv);
        rhs.swap(col, piv);
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = mat[r][col] / mat[col][col];
            for c in col..n {
                mat[r][c] -= factor * mat[col][c];
            }
            rhs[r] -= factor * rhs[col];
        }
    }
    Some((0..n).map(|i| rhs[i] / mat[i][i]).collect())
}

// ---------------------------------------------------------------------------
// Variation of parameters (second order): yp = −y1∫(y2 g/W) + y2∫(y1 g/W)
// where the equation is y'' + P y' + Q y = g, W = y1 y2' − y2 y1'.
// ---------------------------------------------------------------------------

fn variation_of_parameters(
    input: &OdeInput,
    coeffs: &[ExprId],
    basis: &[ExprId],
    r: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, DsolveError> {
    let x = input.x;
    let (y1, y2) = (basis[0], basis[1]);
    // Normalise: divide through by leading coefficient a2.
    let a2 = coeffs[2];
    let g = super::div(r, a2, pool); // g = r/a2
    let y1p = ddx(y1, x, pool)?;
    let y2p = ddx(y2, x, pool)?;
    // Wronskian W = y1 y2' − y2 y1'
    let w = sub(
        simp(pool.mul(vec![y1, y2p]), pool),
        simp(pool.mul(vec![y2, y1p]), pool),
        pool,
    );
    if is_zero(w, pool) {
        return Ok(None);
    }
    let int1 = integrate_or_decline(
        super::div(simp(pool.mul(vec![y2, g]), pool), w, pool),
        x,
        pool,
    )?;
    let int2 = integrate_or_decline(
        super::div(simp(pool.mul(vec![y1, g]), pool), w, pool),
        x,
        pool,
    )?;
    let term1 = pool.mul(vec![pool.integer(-1_i32), y1, int1]);
    let term2 = pool.mul(vec![y2, int2]);
    let yp = simp(pool.add(vec![term1, term2]), pool);
    Ok(Some(yp))
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
) -> Result<Option<DsolveResult>, DsolveError> {
    let x = input.x;
    // Require coeffs[k] = c_k · x^k with constant c_k.
    let mut c = Vec::with_capacity(coeffs.len());
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
        let v = try_expr_f64(ck, pool)
            .ok_or_else(|| DsolveError::Unsupported("non-numeric Euler coefficient".to_string()))?;
        c.push(v);
    }
    // Only handle homogeneous Euler–Cauchy for now.
    if !is_zero(r, pool) {
        return Ok(None);
    }
    // Second order: a m(m−1) + b m + c0 = 0 → a m² + (b−a) m + c0 = 0.
    if c.len() != 3 {
        return Ok(None);
    }
    let (a, b, c0) = (c[2], c[1], c[0]);
    if a == 0.0 {
        return Ok(None);
    }
    // indicial: a m² + (b − a) m + c0
    let disc = (b - a) * (b - a) - 4.0 * a * c0;
    let c1 = gen.fresh(pool);
    let c2 = gen.fresh(pool);
    let y_expr = if disc > 1e-10 {
        let s = disc.sqrt();
        let m1 = (-(b - a) + s) / (2.0 * a);
        let m2 = (-(b - a) - s) / (2.0 * a);
        let xm1 = pow_real(x, m1, pool);
        let xm2 = pow_real(x, m2, pool);
        simp(
            pool.add(vec![pool.mul(vec![c1, xm1]), pool.mul(vec![c2, xm2])]),
            pool,
        )
    } else if disc.abs() <= 1e-10 {
        // repeated root m: y = (C1 + C2 log x) x^m
        let m = -(b - a) / (2.0 * a);
        let xm = pow_real(x, m, pool);
        let logx = pool.func("log", vec![x]);
        let inner = pool.add(vec![c1, pool.mul(vec![c2, logx])]);
        simp(pool.mul(vec![inner, xm]), pool)
    } else {
        // complex roots m = α ± βi: y = x^α (C1 cos(β log x) + C2 sin(β log x))
        let alpha = -(b - a) / (2.0 * a);
        let beta = (-disc).sqrt() / (2.0 * a);
        let xa = pow_real(x, alpha, pool);
        let logx = pool.func("log", vec![x]);
        let blogx = mul_const(beta, logx, pool);
        let cos = pool.func("cos", vec![blogx]);
        let sin = pool.func("sin", vec![blogx]);
        let inner = pool.add(vec![pool.mul(vec![c1, cos]), pool.mul(vec![c2, sin])]);
        simp(pool.mul(vec![xa, inner]), pool)
    };
    let constants = vec![c1, c2];
    match residual_is_zero(input, y_expr, &constants, pool) {
        Ok(()) => Ok(Some(DsolveResult {
            solutions: vec![DsolveSolution {
                y_of_x: y_expr,
                constants,
                method: "euler_cauchy",
            }],
        })),
        Err(_) => Ok(None),
    }
}

fn pow_real(x: ExprId, m: f64, pool: &ExprPool) -> ExprId {
    let me = f64_to_expr(m, pool);
    simp(pool.pow(x, me), pool)
}
