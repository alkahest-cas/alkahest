//! V2-17 — Eigenvalues, eigenvectors, and diagonalization for dense symbolic matrices
//! whose characteristic polynomial splits over ℚ into linear, quadratic, or cubic
//! factors.
//!
//! The characteristic polynomial is `det(λI − M)` in a fresh λ symbol. Entries may
//! be rational; the determinant is read as a ℚ-polynomial in λ and cleared to a
//! ℤ-polynomial for factorization (same roots). Irreducible cubics are solved via
//! the trigonometric (casus irreducibilis) or Cardano formula.

#![allow(clippy::needless_range_loop)]

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::matrix::Matrix;
use crate::poly::error::ConversionError;
use crate::poly::unipoly::UniPoly;
use crate::poly::{factor_univariate_z, FactorError};
use crate::simplify::engine::{simplify, simplify_expanded};
use rug::Rational;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Errors from eigen-decomposition helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EigenError {
    /// `eigenvals` requires a square matrix.
    NonSquare,
    /// The determinant polynomial could not be read as ℤ\[λ\].
    CharPolyConversion(ConversionError),
    /// FLINT factorization failed.
    Factorization(FactorError),
    /// The characteristic polynomial has an irreducible factor of degree greater than three.
    UnsupportedIrreducibleDegree { degree: usize },
    /// Algebraic and geometric multiplicity disagree (Jordan block situation).
    NonDiagonalizable,
    /// Gaussian elimination failed to produce a numerical field representation.
    KernelComputationFailed,
    /// `P` in `diagonalize` is singular / not invertible.
    SingularModalMatrix,
}

impl fmt::Display for EigenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EigenError::NonSquare => write!(f, "eigen decomposition requires a square matrix"),
            EigenError::CharPolyConversion(e) => write!(f, "characteristic polynomial: {e}"),
            EigenError::Factorization(e) => write!(f, "factorization failed: {e}"),
            EigenError::UnsupportedIrreducibleDegree { degree } => write!(
                f,
                "irreducible characteristic factor of degree {degree}; only degrees 1–3 are supported"
            ),
            EigenError::NonDiagonalizable => {
                write!(f, "matrix is not diagonalizable over the computed eigenbasis")
            }
            EigenError::KernelComputationFailed => write!(
                f,
                "could not compute eigenvectors (nullspace) for this coefficient field"
            ),
            EigenError::SingularModalMatrix => {
                write!(f, "eigenvector matrix is singular — no diagonal decomposition")
            }
        }
    }
}

impl std::error::Error for EigenError {}

impl crate::errors::AlkahestError for EigenError {
    fn code(&self) -> &'static str {
        match self {
            EigenError::NonSquare => "E-EIGEN-001",
            EigenError::CharPolyConversion(_) => "E-EIGEN-002",
            EigenError::Factorization(_) => "E-EIGEN-003",
            EigenError::UnsupportedIrreducibleDegree { .. } => "E-EIGEN-004",
            EigenError::NonDiagonalizable => "E-EIGEN-005",
            EigenError::KernelComputationFailed => "E-EIGEN-006",
            EigenError::SingularModalMatrix => "E-EIGEN-007",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            EigenError::NonSquare => Some("pass a square n×n matrix"),
            EigenError::CharPolyConversion(_) => Some(
                "entries must simplify to rationals/constants so det(λI−M) is a polynomial in λ",
            ),
            EigenError::Factorization(_) => None,
            EigenError::UnsupportedIrreducibleDegree { .. } => {
                Some("degree-4+ irreducible characteristic factors require a CAS / algebraic-numbers extension")
            }
            EigenError::NonDiagonalizable => {
                Some("use Jordan-form tooling or restrict to diagonalizable matrices")
            }
            EigenError::KernelComputationFailed => Some(
                "try a purely rational spectrum or a matrix with quadratic splitting only over ℚ",
            ),
            EigenError::SingularModalMatrix => Some(
                "the computed eigenvectors are linearly dependent; check multiplicities or input",
            ),
        }
    }
}

static EIGEN_LAM_SEQ: AtomicUsize = AtomicUsize::new(0);

fn fresh_lambda(pool: &ExprPool) -> ExprId {
    let n = EIGEN_LAM_SEQ.fetch_add(1, Ordering::Relaxed);
    pool.symbol(format!("__eigen_lambda_{n}"), Domain::Complex)
}

/// √(−1), used internally for quadratic splitting roots and ℚ(\`i\`) nullspaces.
pub(crate) fn imag_unit(pool: &ExprPool) -> ExprId {
    let neg_one = pool.integer(-1_i32);
    pool.func("sqrt", vec![neg_one])
}

/// `(det(λ I − M), λ)` — the determinant is simplified before return.
pub fn characteristic_polynomial_lambda_minus_m(
    m: &Matrix,
    pool: &ExprPool,
) -> Result<(ExprId, ExprId), EigenError> {
    if m.rows != m.cols {
        return Err(EigenError::NonSquare);
    }
    let lam = fresh_lambda(pool);
    let lm = lambda_identity_minus_m(m, lam, pool);
    let det = lm.det(pool).map_err(|_| EigenError::NonSquare)?;
    Ok((simplify(det, pool).value, lam))
}

/// multiset of eigenvalue Expr → algebraic multiplicity
pub fn eigenvalues(m: &Matrix, pool: &ExprPool) -> Result<Vec<(ExprId, usize)>, EigenError> {
    let (poly_e, lam) = characteristic_polynomial_lambda_minus_m(m, pool)?;
    match eigenvalues_from_char_poly(poly_e, lam, pool) {
        Ok(v) => Ok(v),
        Err(EigenError::CharPolyConversion(_)) => {
            // The characteristic polynomial has non-rational (free-symbol) coefficients,
            // so it cannot be cleared to ℤ[λ]. For 2×2 matrices the eigenvalues are still
            // available in closed form via the quadratic formula, fully symbolically.
            symbolic_eigenvalues_2x2(m, pool)
        }
        Err(e) => Err(e),
    }
}

/// Closed-form eigenvalues of a 2×2 matrix `[[a, b], [c, d]]` over arbitrary symbolic
/// entries: roots of `λ² − (a+d)λ + (ad − bc)`. Returns the two roots
/// `(tr ± √(tr² − 4·det)) / 2`, collapsing to a single eigenvalue of multiplicity 2
/// when the discriminant simplifies to exactly zero.
///
/// This deliberately covers only the 2×2 case — higher-dimensional symbolic spectra
/// need general radical/algebraic-number machinery and are out of scope here.
fn symbolic_eigenvalues_2x2(
    m: &Matrix,
    pool: &ExprPool,
) -> Result<Vec<(ExprId, usize)>, EigenError> {
    if m.rows != m.cols {
        return Err(EigenError::NonSquare);
    }
    if m.rows != 2 {
        // No general symbolic spectrum for n ≥ 3 yet.
        return Err(EigenError::UnsupportedIrreducibleDegree { degree: m.rows });
    }
    let a = m.get(0, 0);
    let b = m.get(0, 1);
    let c = m.get(1, 0);
    let d = m.get(1, 1);
    let neg_one = pool.integer(-1_i32);

    // trace = a + d
    let trace = simplify(pool.add(vec![a, d]), pool).value;
    // det = a*d - b*c
    let ad = pool.mul(vec![a, d]);
    let bc = pool.mul(vec![b, c]);
    let det = simplify(pool.add(vec![ad, pool.mul(vec![neg_one, bc])]), pool).value;

    // discriminant = trace² - 4*det
    let trace_sq = pool.pow(trace, pool.integer(2_i32));
    let four_det = pool.mul(vec![pool.integer(4_i32), det]);
    let disc = simplify(
        pool.add(vec![trace_sq, pool.mul(vec![neg_one, four_det])]),
        pool,
    )
    .value;

    let half = pool.rational(rug::Integer::from(1), rug::Integer::from(2));

    // Repeated root when the discriminant is exactly zero: λ = trace/2 (mult 2).
    if expr_is_exactly_zero(pool, disc) {
        let lam = simplify(pool.mul(vec![half, trace]), pool).value;
        return Ok(vec![(lam, 2)]);
    }

    let sqrt_disc = simplify(pool.func("sqrt", vec![disc]), pool).value;
    let neg_sqrt = simplify(pool.mul(vec![neg_one, sqrt_disc]), pool).value;
    let lam_plus = simplify(
        pool.mul(vec![
            half,
            simplify(pool.add(vec![trace, sqrt_disc]), pool).value,
        ]),
        pool,
    )
    .value;
    let lam_minus = simplify(
        pool.mul(vec![
            half,
            simplify(pool.add(vec![trace, neg_sqrt]), pool).value,
        ]),
        pool,
    )
    .value;

    // If the two roots collapse after simplification (e.g. discriminant simplified to a
    // perfect square that the zero-check missed), treat as a repeated eigenvalue.
    if lam_plus == lam_minus {
        return Ok(vec![(lam_plus, 2)]);
    }
    let (x, y) = order_two_roots(lam_plus, lam_minus, pool);
    Ok(vec![(x, 1), (y, 1)])
}

/// `(value, multiplicity, column eigenvectors)`
pub fn eigenvectors(
    m: &Matrix,
    pool: &ExprPool,
) -> Result<Vec<(ExprId, usize, Vec<Matrix>)>, EigenError> {
    let vals = eigenvalues(m, pool)?;
    let mut out = Vec::with_capacity(vals.len());
    for (lambda, mult) in vals {
        let b = m_minus_lambda_scaled(m, lambda, pool);
        let vecs =
            kernel_column_basis(&b, pool).map_err(|_| EigenError::KernelComputationFailed)?;
        out.push((lambda, mult, vecs));
    }
    Ok(out)
}

/// Returns `(P, D)` with `M·P == P·D` (same convention as SymPy: columns of `P` are eigenvectors).
pub fn diagonalize(m: &Matrix, pool: &ExprPool) -> Result<(Matrix, Matrix), EigenError> {
    let evecs = eigenvectors(m, pool)?;
    for (_lambda, alg_m, vecs) in &evecs {
        if vecs.len() != *alg_m {
            return Err(EigenError::NonDiagonalizable);
        }
    }
    let n = m.rows;
    let mut cols: Vec<Matrix> = Vec::with_capacity(n);
    let mut diag_entries: Vec<ExprId> = Vec::with_capacity(n);
    for (lambda, _alg_m, vecs) in evecs {
        for v in vecs {
            cols.push(v);
            diag_entries.push(lambda);
        }
    }
    if cols.len() != n {
        return Err(EigenError::NonDiagonalizable);
    }
    let p_mat =
        concatenate_columns(&cols, pool).map_err(|_| EigenError::KernelComputationFailed)?;
    // Verify full rank geometrically via det / invertibility later
    let d_mat = diagonal_from_entries(&diag_entries, pool);
    if !columns_match_eigen_relation(m, &p_mat, pool, &diag_entries) {
        return Err(EigenError::NonDiagonalizable);
    }
    Ok((p_mat, d_mat))
}

// ---------------------------------------------------------------------------
// Characteristic polynomial → eigenvalues
// ---------------------------------------------------------------------------

fn eigenvalues_from_char_poly(
    poly_e: ExprId,
    lam: ExprId,
    pool: &ExprPool,
) -> Result<Vec<(ExprId, usize)>, EigenError> {
    let uni = UniPoly::from_symbolic_clear_denoms(poly_e, lam, pool)
        .map_err(EigenError::CharPolyConversion)?;
    let fac = factor_univariate_z(&uni).map_err(EigenError::Factorization)?;
    let mut pairs: Vec<(ExprId, usize)> = Vec::new();
    for (base, exp) in fac.factors {
        let d = base.degree() as usize;
        match d {
            0 => continue,
            1 => {
                let root = linear_root(&base, pool)
                    .ok_or(EigenError::UnsupportedIrreducibleDegree { degree: d })?;
                pairs.push((root, exp as usize));
            }
            2 => {
                let (r1, r2) = quadratic_roots(&base, pool)?;
                pairs.push((r1, exp as usize));
                pairs.push((r2, exp as usize));
            }
            3 => {
                for r in cubic_roots(&base, pool)? {
                    pairs.push((r, exp as usize));
                }
            }
            _ => return Err(EigenError::UnsupportedIrreducibleDegree { degree: d }),
        }
    }
    sort_eigenpairs(&pairs, pool)
}

fn linear_root(p: &UniPoly, pool: &ExprPool) -> Option<ExprId> {
    let c = p.coefficients();
    if c.len() != 2 {
        return None;
    }
    // c0 + c1 λ
    let numer = -&c[0];
    let denom = c[1].clone();
    if denom == 0 {
        None
    } else {
        Some(pool.rational(numer, denom))
    }
}

fn quadratic_roots(p: &UniPoly, pool: &ExprPool) -> Result<(ExprId, ExprId), EigenError> {
    let c = p.coefficients();
    if c.len() != 3 {
        return Err(EigenError::UnsupportedIrreducibleDegree {
            degree: p.degree().max(0) as usize,
        });
    }
    let c0 = c[0].clone();
    let c1 = c[1].clone();
    let c2 = c[2].clone();
    if c2 == 0 {
        return Err(EigenError::UnsupportedIrreducibleDegree { degree: 1 });
    }
    // Prefer ±√(-c₀/c₂) when c₁ = 0 (e.g. λ² + 1) so roots use √(−1) and match the ℚ(i) path.
    if c1 == 0 {
        let r_rat = Rational::from((rug::Integer::from(0) - &c0, c2.clone()));
        let inner_sqrt = if r_rat.denom().clone() == 1 {
            pool.integer(r_rat.numer().clone())
        } else {
            pool.rational(r_rat.numer().clone(), r_rat.denom().clone())
        };
        let sd = simplify(pool.func("sqrt", vec![inner_sqrt]), pool).value;
        let r_minus = simplify(pool.mul(vec![pool.integer(-1_i32), sd]), pool).value;
        let (x, y) = order_two_roots(sd, r_minus, pool);
        return Ok((x, y));
    }
    let mut disc = c1.clone() * c1.clone();
    disc -= rug::Integer::from(4) * c0 * c2.clone();
    let sqrt_d = simplify(
        pool.func("sqrt", vec![expr_from_integer(&disc, pool)]),
        pool,
    )
    .value;
    let neg_c1 = rug::Integer::from(0) - &c1;
    let minus_c1 = expr_from_integer(&neg_c1, pool);
    let two_a = c2.clone() * rug::Integer::from(2);
    let inv_2a = pool.rational(rug::Integer::from(1), two_a.clone());
    let r_plus = simplify(
        pool.mul(vec![
            inv_2a,
            simplify(pool.add(vec![minus_c1, sqrt_d]), pool).value,
        ]),
        pool,
    )
    .value;
    let neg_sqrt = simplify(pool.mul(vec![pool.integer(-1_i32), sqrt_d]), pool).value;
    let r_minus = simplify(
        pool.mul(vec![
            inv_2a,
            simplify(pool.add(vec![minus_c1, neg_sqrt]), pool).value,
        ]),
        pool,
    )
    .value;
    let (x, y) = order_two_roots(r_plus, r_minus, pool);
    Ok((x, y))
}

fn expr_from_integer(n: &rug::Integer, pool: &ExprPool) -> ExprId {
    pool.integer(n.clone())
}

/// Lexicographic sort on `pool.display` for stable tests.
fn sort_eigenpairs(
    pairs: &[(ExprId, usize)],
    pool: &ExprPool,
) -> Result<Vec<(ExprId, usize)>, EigenError> {
    let mut v: Vec<(ExprId, usize)> = pairs.to_vec();
    v.sort_by(|a, b| {
        pool.display(a.0)
            .to_string()
            .cmp(&pool.display(b.0).to_string())
    });
    Ok(v)
}

fn order_two_roots(a: ExprId, b: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let sa = pool.display(a).to_string();
    let sb = pool.display(b).to_string();
    if sa <= sb {
        (a, b)
    } else {
        (b, a)
    }
}

fn rational_to_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    if r.denom().clone() == 1 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

/// Roots of an irreducible cubic `c₀ + c₁λ + c₂λ² + c₃λ³` via depression +
/// trigonometric Cardano (three real roots) or radical Cardano (one real root).
fn cubic_roots(p: &UniPoly, pool: &ExprPool) -> Result<[ExprId; 3], EigenError> {
    let c = p.coefficients();
    if c.len() != 4 {
        return Err(EigenError::UnsupportedIrreducibleDegree {
            degree: p.degree().max(0) as usize,
        });
    }
    let c0 = c[0].clone();
    let c1 = c[1].clone();
    let c2 = c[2].clone();
    let c3 = c[3].clone();
    if c3 == 0 {
        return Err(EigenError::UnsupportedIrreducibleDegree { degree: 2 });
    }
    // Monic coefficients: λ³ + a λ² + b λ + c = 0.
    let a = Rational::from((c2, c3.clone()));
    let b = Rational::from((c1, c3.clone()));
    let cc = Rational::from((c0, c3));
    // Depress: λ = t − a/3,   t³ + p t + q = 0.
    let a2 = a.clone() * a.clone();
    let a3 = a2.clone() * a.clone();
    let p_coeff = b.clone() - a2 / Rational::from(3);
    let q_coeff =
        cc + (Rational::from(2) * a3 - Rational::from(9) * a.clone() * b) / Rational::from(27);
    let shift = rational_to_expr(&(a / Rational::from(3)), pool);
    let neg_shift = simplify(pool.mul(vec![pool.integer(-1_i32), shift]), pool).value;

    let half_q = q_coeff.clone() / Rational::from(2);
    let third_p = p_coeff.clone() / Rational::from(3);
    // Δ = (q/2)² + (p/3)³.  Δ < 0 ⇒ three distinct real roots (casus irreducibilis).
    let mut delta = half_q.clone() * half_q.clone();
    delta += third_p.clone() * third_p.clone() * third_p.clone();

    let roots_t: [ExprId; 3] = if delta < 0 {
        // t_k = 2√(−p/3) cos((θ + 2πk)/3),  cos θ = (−q/2) / (√(−p/3))³.
        let neg_third_p = Rational::from(0) - third_p;
        let sqrt_neg_third_p = simplify(
            pool.func("sqrt", vec![rational_to_expr(&neg_third_p, pool)]),
            pool,
        )
        .value;
        let two_sqrt = simplify(pool.mul(vec![pool.integer(2_i32), sqrt_neg_third_p]), pool).value;
        let denom = simplify(pool.pow(sqrt_neg_third_p, pool.integer(3_i32)), pool).value;
        let neg_half_q = rational_to_expr(&(Rational::from(0) - half_q), pool);
        let cos_theta = simplify(
            pool.mul(vec![neg_half_q, pool.pow(denom, pool.integer(-1_i32))]),
            pool,
        )
        .value;
        let theta = pool.func("acos", vec![cos_theta]);
        let pi = pool.symbol("pi", Domain::Real);
        let two_pi = pool.mul(vec![pool.integer(2_i32), pi]);
        let mut out = [pool.integer(0_i32); 3];
        for k in 0..3 {
            let angle = simplify(
                pool.mul(vec![
                    pool.rational(rug::Integer::from(1), rug::Integer::from(3)),
                    pool.add(vec![theta, pool.mul(vec![two_pi, pool.integer(k as i32)])]),
                ]),
                pool,
            )
            .value;
            out[k] = simplify(
                pool.mul(vec![two_sqrt, pool.func("cos", vec![angle])]),
                pool,
            )
            .value;
        }
        out
    } else {
        // One real root via Cardano: A = ∛(−q/2 + √Δ), B = ∛(−q/2 − √Δ), t₀ = A+B.
        let sqrt_delta = simplify(
            pool.func("sqrt", vec![rational_to_expr(&delta, pool)]),
            pool,
        )
        .value;
        let neg_half_q = rational_to_expr(&(Rational::from(0) - half_q), pool);
        let a_cbrt = simplify(
            pool.pow(
                pool.add(vec![neg_half_q, sqrt_delta]),
                pool.rational(rug::Integer::from(1), rug::Integer::from(3)),
            ),
            pool,
        )
        .value;
        let b_cbrt = simplify(
            pool.pow(
                pool.add(vec![
                    neg_half_q,
                    pool.mul(vec![pool.integer(-1_i32), sqrt_delta]),
                ]),
                pool.rational(rug::Integer::from(1), rug::Integer::from(3)),
            ),
            pool,
        )
        .value;
        let t0 = simplify(pool.add(vec![a_cbrt, b_cbrt]), pool).value;
        // Complex conjugate pair: −(A+B)/2 ± i √3 (A−B)/2.
        let half = pool.rational(rug::Integer::from(1), rug::Integer::from(2));
        let neg_half_sum = simplify(pool.mul(vec![pool.integer(-1_i32), half, t0]), pool).value;
        let aminus_b = simplify(
            pool.add(vec![a_cbrt, pool.mul(vec![pool.integer(-1_i32), b_cbrt])]),
            pool,
        )
        .value;
        let imag = simplify(
            pool.mul(vec![
                half,
                pool.func("sqrt", vec![pool.integer(3_i32)]),
                aminus_b,
                imag_unit(pool),
            ]),
            pool,
        )
        .value;
        let t1 = simplify(pool.add(vec![neg_half_sum, imag]), pool).value;
        let t2 = simplify(
            pool.add(vec![
                neg_half_sum,
                pool.mul(vec![pool.integer(-1_i32), imag]),
            ]),
            pool,
        )
        .value;
        [t0, t1, t2]
    };

    let mut out = [pool.integer(0_i32); 3];
    for (i, t) in roots_t.into_iter().enumerate() {
        out[i] = simplify(pool.add(vec![t, neg_shift]), pool).value;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// λ I − M and M − λ I
// ---------------------------------------------------------------------------

fn lambda_identity_minus_m(m: &Matrix, lam: ExprId, pool: &ExprPool) -> Matrix {
    let n = m.rows;
    let neg_one = pool.integer(-1_i32);
    let mut data = Vec::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            let e = if r == c {
                let term = pool.mul(vec![neg_one, m.get(r, c)]);
                pool.add(vec![lam, term])
            } else {
                pool.mul(vec![neg_one, m.get(r, c)])
            };
            data.push(e);
        }
    }
    Matrix {
        data,
        rows: n,
        cols: n,
    }
}

pub(crate) fn m_minus_lambda_scaled(m: &Matrix, lambda: ExprId, pool: &ExprPool) -> Matrix {
    let n = m.rows;
    let mut data = Vec::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            let e = if r == c {
                let neg_l = pool.mul(vec![pool.integer(-1_i32), lambda]);
                pool.add(vec![m.get(r, c), neg_l])
            } else {
                m.get(r, c)
            };
            data.push(e);
        }
    }
    Matrix {
        data,
        rows: n,
        cols: n,
    }
}

// ---------------------------------------------------------------------------
// Nullspace
// ---------------------------------------------------------------------------

fn kernel_2x2_column_basis(m: &Matrix, pool: &ExprPool) -> Option<Vec<Matrix>> {
    let a00 = simplify(m.get(0, 0), pool).value;
    let b01 = simplify(m.get(0, 1), pool).value;
    let c10 = simplify(m.get(1, 0), pool).value;
    let d11 = simplify(m.get(1, 1), pool).value;
    // Full-rank gate for numeric/rational matrices: if det is a nonzero
    // constant then the kernel is trivial.  Do *not* use an `M·v = 0` check on
    // the candidate perpendicular — for symbolic `(A − λI)` that residual only
    // vanishes after substituting an eigenvalue, so the check would wrongly
    // drop legitimate eigenspace bases.
    let det = simplify(
        pool.add(vec![
            pool.mul(vec![a00, d11]),
            pool.mul(vec![pool.integer(-1_i32), b01, c10]),
        ]),
        pool,
    )
    .value;
    let det_nonzero_const = match pool.get(det) {
        ExprData::Integer(n) => n.0 != 0,
        ExprData::Rational(r) => r.0 != 0,
        _ => false,
    };
    if det_nonzero_const {
        return Some(Vec::new());
    }
    let neg_one = pool.integer(-1_i32);
    let (a, b) = if expr_is_exactly_zero(pool, a00) && expr_is_exactly_zero(pool, b01) {
        if expr_is_exactly_zero(pool, c10) && expr_is_exactly_zero(pool, d11) {
            // Zero matrix — fall through to the general rational/Gauss path,
            // which returns a full 2-dimensional basis.
            return None;
        }
        (c10, d11)
    } else {
        (a00, b01)
    };
    let neg_b = simplify(pool.mul(vec![neg_one, b]), pool).value;
    let col = Matrix::new(vec![vec![neg_b], vec![a]]).ok()?;
    Some(vec![col])
}

pub(crate) fn kernel_column_basis(m: &Matrix, pool: &ExprPool) -> Result<Vec<Matrix>, ()> {
    if m.rows == 2 && m.cols == 2 {
        if let Some(bas) = kernel_2x2_column_basis(m, pool) {
            return Ok(bas);
        }
    }
    let cols = m.cols;
    let n = m.rows;
    if let Some(rm) = matrix_to_rational_grid(m, pool) {
        let bas = rational_nullspace_basis(&rm, n, cols);
        return Ok(bas
            .into_iter()
            .map(|col| col_vector_from_rationals(&col, pool))
            .collect());
    }
    let iu = imag_unit(pool);
    if let Some(qm) = matrix_to_qi_grid(m, iu, pool) {
        let bas = qi_nullspace_basis(&qm, n, cols);
        return Ok(bas
            .into_iter()
            .map(|col| col_vector_from_qi(&col, iu, pool))
            .collect());
    }
    let bas = gauss_nullspace_expr(m, pool)?;
    Ok(bas
        .into_iter()
        .map(|col| col_vector_from_expr_slice(&col, pool))
        .collect())
}

fn col_vector_from_rationals(v: &[Rational], pool: &ExprPool) -> Matrix {
    let rows: Vec<Vec<ExprId>> = v
        .iter()
        .map(|r| vec![pool.rational(r.numer().clone(), r.denom().clone())])
        .collect();
    Matrix::new(rows).expect("cols")
}

fn col_vector_from_qi(v: &[(Rational, Rational)], imag: ExprId, pool: &ExprPool) -> Matrix {
    let rows: Vec<Vec<ExprId>> = v
        .iter()
        .map(|(re, im)| {
            let re_e = pool.rational(re.numer().clone(), re.denom().clone());
            if im == &Rational::from(0) {
                vec![re_e]
            } else {
                let im_e = pool.rational(im.numer().clone(), im.denom().clone());
                let im_term = simplify(pool.mul(vec![im_e, imag]), pool).value;
                vec![simplify(pool.add(vec![re_e, im_term]), pool).value]
            }
        })
        .collect();
    Matrix::new(rows).expect("qi col")
}

fn col_vector_from_expr_slice(v: &[ExprId], _pool: &ExprPool) -> Matrix {
    let rows: Vec<Vec<ExprId>> = v.iter().copied().map(|e| vec![e]).collect();
    Matrix::new(rows).expect("expr col")
}

fn is_sqrt_of_negative_one(pool: &ExprPool, e: ExprId) -> bool {
    match pool.get(e) {
        ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
            args[0] == pool.integer(-1_i32)
        }
        _ => false,
    }
}

fn squash_sqrt_minus_one_squared(e: ExprId, pool: &ExprPool) -> ExprId {
    fn rec(e: ExprId, pool: &ExprPool) -> ExprId {
        match pool.get(e) {
            ExprData::Pow { base, exp } => {
                if let ExprData::Integer(n) = pool.get(exp) {
                    if n.0 == 2 && is_sqrt_of_negative_one(pool, base) {
                        return pool.integer(-1_i32);
                    }
                }
                let nb = rec(base, pool);
                let ne = rec(exp, pool);
                pool.pow(nb, ne)
            }
            ExprData::Add(args) => {
                let v: Vec<ExprId> = args.iter().map(|&a| rec(a, pool)).collect();
                pool.add(v)
            }
            ExprData::Mul(args) => {
                let v: Vec<ExprId> = args.iter().map(|&a| rec(a, pool)).collect();
                pool.mul(v)
            }
            _ => e,
        }
    }
    rec(e, pool)
}

fn deep_normalize_for_compare(expr: ExprId, pool: &ExprPool, rounds: usize) -> ExprId {
    let mut cur = squash_sqrt_minus_one_squared(expr, pool);
    for _ in 0..rounds {
        let n = simplify_expanded(cur, pool).value;
        let n2 = simplify(n, pool).value;
        if n2 == cur {
            break;
        }
        cur = n2;
    }
    cur
}

#[allow(dead_code)]
pub(crate) fn matrix_eq_simplified(a: &Matrix, b: &Matrix, pool: &ExprPool) -> bool {
    if a.rows != b.rows || a.cols != b.cols {
        return false;
    }
    for i in 0..a.rows * a.cols {
        let ea = deep_normalize_for_compare(a.entries()[i], pool, 12);
        let eb = deep_normalize_for_compare(b.entries()[i], pool, 12);
        if ea != eb {
            return false;
        }
    }
    true
}

/// `(M P)_{r,j} == λ_j P_{r,j}` for all `r`, `j`.
fn columns_match_eigen_relation(
    m: &Matrix,
    p: &Matrix,
    pool: &ExprPool,
    lambdas: &[ExprId],
) -> bool {
    let n = m.rows;
    if m.cols != n || p.rows != n || p.cols != n || lambdas.len() != n {
        return false;
    }
    for j in 0..n {
        let lam = lambdas[j];
        for r in 0..n {
            let mut terms: Vec<ExprId> = Vec::with_capacity(n);
            for k in 0..n {
                terms.push(pool.mul(vec![m.get(r, k), p.get(k, j)]));
            }
            let lhs = simplify(pool.add(terms), pool).value;
            let rhs = simplify(pool.mul(vec![lam, p.get(r, j)]), pool).value;
            let lhs3 = deep_normalize_for_compare(lhs, pool, 12);
            let rhs3 = deep_normalize_for_compare(rhs, pool, 12);
            if lhs3 != rhs3 {
                return false;
            }
        }
    }
    true
}

// --- ℚ grid ---

fn matrix_to_rational_grid(m: &Matrix, pool: &ExprPool) -> Option<Vec<Vec<Rational>>> {
    let mut g = Vec::with_capacity(m.rows);
    for r in 0..m.rows {
        let mut row = Vec::with_capacity(m.cols);
        for c in 0..m.cols {
            row.push(expr_to_rational_strict(m.get(r, c), pool)?);
        }
        g.push(row);
    }
    Some(g)
}

fn expr_to_rational_strict(e: ExprId, pool: &ExprPool) -> Option<Rational> {
    match pool.get(e) {
        ExprData::Integer(ref n) => Some(Rational::from((n.0.clone(), rug::Integer::from(1)))),
        ExprData::Rational(ref r) => Some(r.0.clone()),
        ExprData::Add(ref args) => {
            let mut acc = Rational::from(0);
            for &a in args {
                acc += expr_to_rational_strict(a, pool)?;
            }
            Some(acc)
        }
        ExprData::Mul(ref args) => {
            let mut acc = Rational::from(1);
            for &a in args {
                acc *= expr_to_rational_strict(a, pool)?;
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(n) => {
                let ei = n.0.to_i32()?;
                if ei < 0 {
                    None
                } else {
                    let b = expr_to_rational_strict(base, pool)?;
                    Some(if ei == 0 {
                        Rational::from(1)
                    } else {
                        let mut acc = Rational::from(1);
                        for _ in 0..ei {
                            acc *= b.clone();
                        }
                        acc
                    })
                }
            }
            _ => None,
        },
        _ => None,
    }
}

fn rational_nullspace_basis(mat: &[Vec<Rational>], rows: usize, cols: usize) -> Vec<Vec<Rational>> {
    let mut a = mat.to_vec();
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut r = 0usize;
    for c in 0..cols {
        if r >= rows {
            break;
        }
        let mut piv = None;
        for rr in r..rows {
            if a[rr][c] != 0 {
                piv = Some(rr);
                break;
            }
        }
        let Some(pr) = piv else { continue };
        if pr != r {
            a.swap(pr, r);
        }
        let inv = Rational::from(1) / a[r][c].clone();
        for cc in 0..cols {
            a[r][cc] *= inv.clone();
        }
        for rr in 0..rows {
            if rr == r {
                continue;
            }
            let f = a[rr][c].clone();
            if f == 0 {
                continue;
            }
            for cc in 0..cols {
                let sub = f.clone() * a[r][cc].clone();
                a[rr][cc] -= sub;
            }
        }
        pivot_cols.push(c);
        r += 1;
    }
    let mut is_pivot = vec![false; cols];
    for &pc in &pivot_cols {
        is_pivot[pc] = true;
    }
    let mut bases: Vec<Vec<Rational>> = Vec::new();
    for fc in 0..cols {
        if is_pivot[fc] {
            continue;
        }
        let mut v = vec![Rational::from(0); cols];
        v[fc] = Rational::from(1);
        for (i, &pc) in pivot_cols.iter().enumerate() {
            if i >= r {
                break;
            }
            v[pc] = -a[i][fc].clone();
        }
        bases.push(v);
    }
    bases
}

// --- ℚ(i) when entries are `(re) + (im)*sqrt(-1)` with rational re, im ---

fn matrix_to_qi_grid(
    m: &Matrix,
    imag: ExprId,
    pool: &ExprPool,
) -> Option<Vec<Vec<(Rational, Rational)>>> {
    let mut g = Vec::with_capacity(m.rows);
    for r in 0..m.rows {
        let mut row = Vec::with_capacity(m.cols);
        for c in 0..m.cols {
            row.push(split_qi_linear(m.get(r, c), imag, pool)?);
        }
        g.push(row);
    }
    Some(g)
}

fn split_qi_linear(e: ExprId, imag: ExprId, pool: &ExprPool) -> Option<(Rational, Rational)> {
    if let Some(r) = expr_to_rational_strict(e, pool) {
        return Some((r, Rational::from(0)));
    }
    if e == imag {
        return Some((Rational::from(0), Rational::from(1)));
    }
    match pool.get(e) {
        ExprData::Mul(ref args) if args.contains(&imag) => {
            let rest: Vec<ExprId> = args.iter().copied().filter(|&x| x != imag).collect();
            let prod = if rest.is_empty() {
                pool.integer(1_i32)
            } else if rest.len() == 1 {
                rest[0]
            } else {
                pool.mul(rest)
            };
            Some((Rational::from(0), expr_to_rational_strict(prod, pool)?))
        }
        ExprData::Add(ref args) => {
            let mut re = Rational::from(0);
            let mut im = Rational::from(0);
            for &a in args {
                if a == imag {
                    im += Rational::from(1);
                } else if let ExprData::Mul(ms) = pool.get(a) {
                    if ms.contains(&imag) {
                        let rest: Vec<ExprId> = ms.iter().copied().filter(|&x| x != imag).collect();
                        let prod = if rest.is_empty() {
                            pool.integer(1_i32)
                        } else if rest.len() == 1 {
                            rest[0]
                        } else {
                            pool.mul(rest)
                        };
                        im += expr_to_rational_strict(prod, pool)?;
                    } else {
                        re += expr_to_rational_strict(a, pool)?;
                    }
                } else {
                    re += expr_to_rational_strict(a, pool)?;
                }
            }
            Some((re, im))
        }
        _ => None,
    }
}

fn qi_mul(a: (Rational, Rational), b: (Rational, Rational)) -> (Rational, Rational) {
    let (ar, ai) = a;
    let (br, bi) = b;
    (
        ar.clone() * br.clone() - ai.clone() * bi.clone(),
        ar * bi + ai * br,
    )
}

fn qi_add(a: (Rational, Rational), b: (Rational, Rational)) -> (Rational, Rational) {
    (a.0 + b.0, a.1 + b.1)
}

fn qi_neg(a: (Rational, Rational)) -> (Rational, Rational) {
    (-a.0, -a.1)
}

fn qi_is_zero(q: &(Rational, Rational)) -> bool {
    q.0.is_zero() && q.1.is_zero()
}

fn qi_inv(a: (Rational, Rational)) -> Option<(Rational, Rational)> {
    let norm = a.0.clone() * a.0.clone() + a.1.clone() * a.1.clone();
    if norm.is_zero() {
        None
    } else {
        Some((a.0.clone() / norm.clone(), (-a.1.clone()) / norm.clone()))
    }
}

fn qi_nullspace_basis(
    mat: &[Vec<(Rational, Rational)>],
    rows: usize,
    cols: usize,
) -> Vec<Vec<(Rational, Rational)>> {
    let mut a = mat.to_vec();
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut r = 0usize;
    for c in 0..cols {
        if r >= rows {
            break;
        }
        let mut piv = None;
        for rr in r..rows {
            if !qi_is_zero(&a[rr][c]) {
                piv = Some(rr);
                break;
            }
        }
        let Some(pr) = piv else { continue };
        if pr != r {
            a.swap(pr, r);
        }
        let inv = qi_inv(a[r][c].clone()).unwrap();
        for cc in 0..cols {
            a[r][cc] = qi_mul(inv.clone(), a[r][cc].clone());
        }
        for rr in 0..rows {
            if rr == r {
                continue;
            }
            let f = a[rr][c].clone();
            if qi_is_zero(&f) {
                continue;
            }
            for cc in 0..cols {
                let sub = qi_mul(f.clone(), a[r][cc].clone());
                a[rr][cc] = qi_add(a[rr][cc].clone(), qi_neg(sub));
            }
        }
        pivot_cols.push(c);
        r += 1;
    }
    let mut is_pivot = vec![false; cols];
    for &pc in &pivot_cols {
        is_pivot[pc] = true;
    }
    let mut bases: Vec<Vec<(Rational, Rational)>> = Vec::new();
    for fc in 0..cols {
        if is_pivot[fc] {
            continue;
        }
        let mut v = vec![(Rational::from(0), Rational::from(0)); cols];
        v[fc] = (Rational::from(1), Rational::from(0));
        for (i, &pc) in pivot_cols.iter().enumerate() {
            if i >= r {
                break;
            }
            v[pc] = qi_neg(a[i][fc].clone());
        }
        bases.push(v);
    }
    bases
}

// --- Expr Gaussian fallback ---

fn gauss_nullspace_expr(m: &Matrix, pool: &ExprPool) -> Result<Vec<Vec<ExprId>>, ()> {
    let rows = m.rows;
    let cols = m.cols;
    let mut a: Vec<Vec<ExprId>> = (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| simplify(m.get(r, c), pool).value)
                .collect()
        })
        .collect();
    let neg_one = pool.integer(-1_i32);
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut r_at = 0usize;
    for c in 0..cols {
        if r_at >= rows {
            break;
        }
        let mut prow = None;
        for rr in r_at..rows {
            let e = simplify(a[rr][c], pool).value;
            if !expr_is_exactly_zero(pool, e) {
                prow = Some((rr, e));
                break;
            }
        }
        let Some((pr, piv)) = prow else { continue };
        if pr != r_at {
            a.swap(pr, r_at);
        }
        let inv_p = simplify(pool.pow(piv, pool.integer(-1_i32)), pool).value;
        for cc in 0..cols {
            a[r_at][cc] = simplify(pool.mul(vec![inv_p, a[r_at][cc]]), pool).value;
        }
        for rr in 0..rows {
            if rr == r_at {
                continue;
            }
            let f = simplify(a[rr][c], pool).value;
            if expr_is_exactly_zero(pool, f) {
                continue;
            }
            for cc in 0..cols {
                let term = simplify(pool.mul(vec![f, a[r_at][cc]]), pool).value;
                let neg_term = simplify(pool.mul(vec![neg_one, term]), pool).value;
                a[rr][cc] = simplify(pool.add(vec![a[rr][cc], neg_term]), pool).value;
            }
        }
        pivot_cols.push(c);
        r_at += 1;
    }
    let mut is_pivot = vec![false; cols];
    for &pc in &pivot_cols {
        is_pivot[pc] = true;
    }
    let mut bases: Vec<Vec<ExprId>> = Vec::new();
    for fc in 0..cols {
        if is_pivot[fc] {
            continue;
        }
        let mut v = vec![pool.integer(0_i32); cols];
        v[fc] = pool.integer(1_i32);
        for (i, &pc) in pivot_cols.iter().enumerate() {
            if i >= r_at {
                break;
            }
            let neg_entry = simplify(pool.mul(vec![neg_one, a[i][fc]]), pool).value;
            v[pc] = neg_entry;
        }
        bases.push(v);
    }
    Ok(bases)
}

fn expr_is_exactly_zero(pool: &ExprPool, e: ExprId) -> bool {
    match pool.get(e) {
        ExprData::Integer(n) => n.0 == 0,
        ExprData::Rational(r) => r.0 == 0,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Column concat + diagonal
// ---------------------------------------------------------------------------

pub(crate) fn concatenate_columns(cols: &[Matrix], _pool: &ExprPool) -> Result<Matrix, ()> {
    if cols.is_empty() {
        return Err(());
    }
    let n = cols[0].rows;
    for c in cols {
        if c.rows != n || c.cols != 1 {
            return Err(());
        }
    }
    let mut data = Vec::with_capacity(n * cols.len());
    for r in 0..n {
        for col in cols {
            data.push(col.get(r, 0));
        }
    }
    Ok(Matrix {
        data,
        rows: n,
        cols: cols.len(),
    })
}

pub(crate) fn diagonal_from_entries(d: &[ExprId], pool: &ExprPool) -> Matrix {
    let n = d.len();
    let mut mat = Matrix::zeros(n, n, pool);
    for i in 0..n {
        mat.set(i, i, d[i]);
    }
    mat
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn jordan_block_eigenspace_one() {
        let p = pool();
        let two = p.integer(2_i32);
        let one = p.integer(1_i32);
        let z = p.integer(0_i32);
        let m = Matrix::new(vec![vec![two, one], vec![z, two]]).unwrap();
        let vals = eigenvalues(&m, &p).unwrap();
        assert_eq!(vals.len(), 1);
        assert_eq!(vals[0].1, 2);
        let vecs = eigenvectors(&m, &p).unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].2.len(), 1);
        assert!(diagonalize(&m, &p).is_err());
    }

    #[test]
    fn similar_rational_three_by_three_eigenvals() {
        // Same shape as SymPy random seed-17 trial — rational similar to an integer diagonal.
        let p = pool();
        let r = |a: i64, b: i64| p.rational(a, b);
        let m = Matrix::new(vec![
            vec![p.integer(2), r(-18, 7), r(-6, 7)],
            vec![p.integer(0), r(12, 7), r(32, 7)],
            vec![p.integer(-2), r(6, 7), r(-26, 7)],
        ])
        .unwrap();
        eigenvalues(&m, &p).unwrap();
    }

    /// Complex number for numeric verification of symbolic eigenvalues.
    #[derive(Clone, Copy)]
    struct C {
        re: f64,
        im: f64,
    }
    impl C {
        fn new(re: f64, im: f64) -> Self {
            C { re, im }
        }
        fn add(self, o: C) -> C {
            C::new(self.re + o.re, self.im + o.im)
        }
        fn mul(self, o: C) -> C {
            C::new(
                self.re * o.re - self.im * o.im,
                self.re * o.im + self.im * o.re,
            )
        }
        fn powi(self, n: i64) -> C {
            if n == 0 {
                return C::new(1.0, 0.0);
            }
            if n < 0 {
                let p = self.powi(-n);
                let den = p.re * p.re + p.im * p.im;
                return C::new(p.re / den, -p.im / den);
            }
            let mut acc = C::new(1.0, 0.0);
            for _ in 0..n {
                acc = acc.mul(self);
            }
            acc
        }
        fn sqrt(self) -> C {
            // Principal square root of a complex number.
            let r = (self.re * self.re + self.im * self.im).sqrt();
            let re = ((r + self.re) / 2.0).max(0.0).sqrt();
            let mut im = ((r - self.re) / 2.0).max(0.0).sqrt();
            if self.im < 0.0 {
                im = -im;
            }
            C::new(re, im)
        }
        fn near(self, o: C) -> bool {
            (self.re - o.re).abs() < 1e-7 && (self.im - o.im).abs() < 1e-7
        }
    }

    /// Numerically evaluate `e` under a symbol→value map (complex), used purely to verify
    /// closed-form symbolic eigenvalues without fighting the simplifier over nested radicals.
    fn eval_c(e: ExprId, env: &[(ExprId, C)], pool: &ExprPool) -> C {
        match pool.get(e) {
            ExprData::Integer(n) => C::new(n.0.to_f64(), 0.0),
            ExprData::Rational(r) => {
                let (num, den) = r.0.clone().into_numer_denom();
                C::new(num.to_f64() / den.to_f64(), 0.0)
            }
            ExprData::Symbol { .. } => env
                .iter()
                .find(|(s, _)| *s == e)
                .map(|(_, v)| *v)
                .unwrap_or(C::new(0.0, 0.0)),
            ExprData::Add(args) => args
                .iter()
                .fold(C::new(0.0, 0.0), |acc, &a| acc.add(eval_c(a, env, pool))),
            ExprData::Mul(args) => args
                .iter()
                .fold(C::new(1.0, 0.0), |acc, &a| acc.mul(eval_c(a, env, pool))),
            ExprData::Pow { base, exp } => {
                let b = eval_c(base, env, pool);
                if let ExprData::Integer(n) = pool.get(exp) {
                    b.powi(n.0.to_i64().unwrap_or(0))
                } else if let ExprData::Rational(r) = pool.get(exp) {
                    // handle ^(1/2) and ^(-1/2) which appear via sqrt lowering
                    let (num, den) = r.0.clone().into_numer_denom();
                    if den == 1 {
                        b.powi(num.to_i64().unwrap_or(0))
                    } else if den == 2 {
                        let s = b.sqrt();
                        s.powi(num.to_i64().unwrap_or(1))
                    } else {
                        C::new(f64::NAN, f64::NAN)
                    }
                } else {
                    C::new(f64::NAN, f64::NAN)
                }
            }
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                eval_c(args[0], env, pool).sqrt()
            }
            _ => C::new(f64::NAN, f64::NAN),
        }
    }

    /// True iff every returned eigenvalue satisfies `λ² − tr·λ + det = 0` for the 2×2 `m`,
    /// checked numerically at several random rational substitutions for the free symbols.
    fn eigvals_satisfy_2x2_charpoly(m: &Matrix, vals: &[(ExprId, usize)], p: &ExprPool) -> bool {
        let a = m.get(0, 0);
        let b = m.get(0, 1);
        let c = m.get(1, 0);
        let d = m.get(1, 1);
        let tr = p.add(vec![a, d]);
        let det = p.add(vec![
            p.mul(vec![a, d]),
            p.mul(vec![p.integer(-1), p.mul(vec![b, c])]),
        ]);
        // Collect free symbols present in the matrix.
        let mut syms: Vec<ExprId> = Vec::new();
        for &e in m.entries() {
            collect_symbols(e, p, &mut syms);
        }
        // A handful of generic substitution points (avoid trivial coincidences).
        let points: [&[f64]; 3] = [
            &[2.0, 3.0, 5.0, 7.0, 11.0, 13.0],
            &[1.5, -2.0, 4.0, 0.5, -3.0, 6.0],
            &[-1.0, 2.0, -3.0, 5.0, 1.0, -4.0],
        ];
        for pt in points.iter() {
            let env: Vec<(ExprId, C)> = syms
                .iter()
                .enumerate()
                .map(|(i, &s)| (s, C::new(pt[i % pt.len()], 0.0)))
                .collect();
            let tr_v = eval_c(tr, &env, p);
            let det_v = eval_c(det, &env, p);
            for (lam, _) in vals {
                let l = eval_c(*lam, &env, p);
                // λ² − tr·λ + det
                let lhs = l.powi(2).add(tr_v.mul(l).mul(C::new(-1.0, 0.0))).add(det_v);
                if !lhs.near(C::new(0.0, 0.0)) {
                    return false;
                }
            }
        }
        true
    }

    fn collect_symbols(e: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
        match pool.get(e) {
            ExprData::Symbol { .. } if !out.contains(&e) => {
                out.push(e);
            }
            ExprData::Add(args) | ExprData::Mul(args) => {
                for a in args.iter() {
                    collect_symbols(*a, pool, out);
                }
            }
            ExprData::Pow { base, exp } => {
                collect_symbols(base, pool, out);
                collect_symbols(exp, pool, out);
            }
            ExprData::Func { args, .. } => {
                for a in args.iter() {
                    collect_symbols(*a, pool, out);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn symbolic_eigenvalues_harmonic_oscillator() {
        // [[0, 1], [-w^2, 0]] — undamped oscillator companion matrix.
        // char poly: λ² + w² → eigenvalues ± w·√(-1)  (i.e. ±iw).
        let p = pool();
        let w = p.symbol("w", Domain::Real);
        let zero = p.integer(0_i32);
        let one = p.integer(1_i32);
        let w2 = p.pow(w, p.integer(2_i32));
        let neg_w2 = p.mul(vec![p.integer(-1_i32), w2]);
        let m = Matrix::new(vec![vec![zero, one], vec![neg_w2, zero]]).unwrap();
        let vals = eigenvalues(&m, &p).unwrap();
        assert_eq!(vals.len(), 2, "two distinct eigenvalues");
        for (_lam, mult) in &vals {
            assert_eq!(*mult, 1);
        }
        // Both satisfy λ² + w² = 0 (verified numerically over the free symbol w).
        assert!(eigvals_satisfy_2x2_charpoly(&m, &vals, &p));
    }

    #[test]
    fn symbolic_eigenvalues_diagonal_free_symbols() {
        // [[a, 0], [0, b]] → eigenvalues solve (λ−a)(λ−b)=0; both roots satisfy char poly.
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let zero = p.integer(0_i32);
        let m = Matrix::new(vec![vec![a, zero], vec![zero, b]]).unwrap();
        let vals = eigenvalues(&m, &p).unwrap();
        assert_eq!(vals.len(), 2);
        assert!(eigvals_satisfy_2x2_charpoly(&m, &vals, &p));
    }

    #[test]
    fn symbolic_eigenvalues_repeated_scalar_matrix() {
        // [[k, 0], [0, k]] → single eigenvalue k with algebraic multiplicity 2.
        let p = pool();
        let k = p.symbol("k", Domain::Real);
        let zero = p.integer(0_i32);
        let m = Matrix::new(vec![vec![k, zero], vec![zero, k]]).unwrap();
        let vals = eigenvalues(&m, &p).unwrap();
        assert_eq!(vals.len(), 1);
        assert_eq!(vals[0].1, 2);
        assert_eq!(vals[0].0, k);
    }

    #[test]
    fn symbolic_eigenvalues_satisfy_char_poly() {
        // General symbolic 2×2 — every returned eigenvalue must satisfy λ²−tr·λ+det = 0.
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let c = p.symbol("c", Domain::Real);
        let d = p.symbol("d", Domain::Real);
        let m = Matrix::new(vec![vec![a, b], vec![c, d]]).unwrap();
        let vals = eigenvalues(&m, &p).unwrap();
        assert_eq!(vals.len(), 2);
        assert!(
            eigvals_satisfy_2x2_charpoly(&m, &vals, &p),
            "eigenvalues do not satisfy the characteristic polynomial"
        );
    }

    #[test]
    fn symbolic_eigenvectors_satisfy_definition() {
        // For a symbolic 2×2 with distinct eigenvalues, each returned eigenvector v must
        // satisfy A·v = λ·v. Verified numerically over random rational substitutions of the
        // free symbol, since the closed-form vectors involve nested radicals.
        let p = pool();
        let w = p.symbol("w", Domain::Real);
        let zero = p.integer(0_i32);
        let one = p.integer(1_i32);
        let w2 = p.pow(w, p.integer(2_i32));
        let neg_w2 = p.mul(vec![p.integer(-1_i32), w2]);
        // [[0, 1], [-w², 0]] — undamped oscillator companion matrix.
        let m = Matrix::new(vec![vec![zero, one], vec![neg_w2, zero]]).unwrap();
        let triples = eigenvectors(&m, &p).unwrap();
        assert_eq!(triples.len(), 2, "two distinct symbolic eigenpairs");

        let a = m.get(0, 0);
        let b = m.get(0, 1);
        let c = m.get(1, 0);
        let d = m.get(1, 1);
        let points: [f64; 3] = [2.0, 3.5, 5.0];
        for (lambda, _mult, vecs) in &triples {
            assert!(!vecs.is_empty(), "eigenvalue must have ≥1 eigenvector");
            for v in vecs {
                let v0 = v.get(0, 0);
                let v1 = v.get(1, 0);
                for &wv in points.iter() {
                    let env = [(w, C::new(wv, 0.0))];
                    let lam = eval_c(*lambda, &env, &p);
                    let x0 = eval_c(v0, &env, &p);
                    let x1 = eval_c(v1, &env, &p);
                    let av0 = eval_c(a, &env, &p).mul(x0).add(eval_c(b, &env, &p).mul(x1));
                    let av1 = eval_c(c, &env, &p).mul(x0).add(eval_c(d, &env, &p).mul(x1));
                    assert!(av0.near(lam.mul(x0)), "row0: A·v ≠ λ·v at w={wv}");
                    assert!(av1.near(lam.mul(x1)), "row1: A·v ≠ λ·v at w={wv}");
                }
            }
        }
    }

    #[test]
    fn rotation_imag_roots() {
        let p = pool();
        let z = p.integer(0_i32);
        let one = p.integer(1_i32);
        let neg_one = p.integer(-1_i32);
        let m = Matrix::new(vec![vec![z, neg_one], vec![one, z]]).unwrap();
        let vals = eigenvalues(&m, &p).unwrap();
        assert_eq!(vals.len(), 2);
        let vecs = eigenvectors(&m, &p).unwrap();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].2.len(), 1);
        assert_eq!(vecs[1].2.len(), 1);
        let (pp, dd) = diagonalize(&m, &p).unwrap();
        let mp = m.mul(&pp, &p).unwrap().simplify_entries(&p);
        let pdd = pp.mul(&dd, &p).unwrap().simplify_entries(&p);
        assert!(matrix_eq_simplified(&mp, &pdd, &p));
    }
}
