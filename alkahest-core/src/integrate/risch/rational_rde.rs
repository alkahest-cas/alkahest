//! Rational Risch Differential Equation (RDE) solver over ℚ(x) (Risch Gap 1).
//!
//! Extends [`super::poly_rde`] from polynomial coefficients to **rational**
//! coefficients.  Solves
//! ```text
//!   v'(x) + f(x)·v(x) = c(x)
//! ```
//! where `f ∈ ℚ[x]` (in the exp tower, `f = k·η'`, always a polynomial) and
//! `c ∈ ℚ(x)` is a rational function, returning `v ∈ ℚ(x)` when one exists.
//!
//! ## Algorithm (Bronstein 2005, §6.1)
//!
//! Because `f` is a polynomial (no poles), a Laurent-series analysis at any pole
//! `α` of `v` shows that `v'` raises the pole order by one while `f·v` keeps it,
//! so the pole order of `v` at `α` is exactly one less than that of `c`.  Hence
//! the denominator of any rational solution divides
//! ```text
//!   E = gcd(B, B')
//! ```
//! where `B` is the (reduced, monic) denominator of `c`.  Writing `v = N/E` for
//! an unknown polynomial `N`, substituting, and clearing denominators yields a
//! **polynomial identity** that is *linear* in the coefficients of `N`:
//! ```text
//!   G·E·N' − G·E'·N + G·E·f·N − C·E = 0,     G = B/E,  c = C/B.
//! ```
//! Equating coefficients gives a linear system over ℚ.  A consistent system
//! certifies an elementary antiderivative `v·exp(kη)`; an inconsistent one
//! certifies that the integral is **non-elementary** (the leftover simple-pole
//! residues are exactly the Ei/Li-type logarithmic part the exp tower cannot
//! express).
//!
//! The homogeneous equation `v' + f·v = 0` has no nonzero rational solution when
//! `f ≠ 0` (its solutions are `const·exp(−∫f)`), so the rational solution is
//! unique — extra unknowns in an over-sized ansatz are forced to zero, and a
//! final substitution check guards against an under-sized degree bound.
//!
//! References:
//!   - Bronstein (2005). *Symbolic Integration I*, §6.1 (RischDE, normal part).
//!   - SymPy `sympy/integrals/rde.py` (`bound_degree`, `spde`, `no_cancel_*`).

use rug::Rational;

use super::poly_rde::{
    degree, poly_add, poly_deriv, poly_mul, poly_one, poly_scale, poly_zero, trim, QPoly,
};

// ---------------------------------------------------------------------------
// Polynomial arithmetic over ℚ not already provided by `poly_rde`
// ---------------------------------------------------------------------------

/// Subtract `b` from `a`.
fn poly_sub(a: &QPoly, b: &QPoly) -> QPoly {
    poly_add(a, &poly_scale(b, &Rational::from(-1)))
}

/// Coefficient of `x^i` (0 outside the stored range or for negative `i`).
fn coeff(p: &QPoly, i: i64) -> Rational {
    if i < 0 {
        return Rational::from(0);
    }
    p.get(i as usize)
        .cloned()
        .unwrap_or_else(|| Rational::from(0))
}

/// Long division over ℚ: returns `(q, r)` with `a = q·b + r`, `deg r < deg b`.
/// `b` must be nonzero.
fn poly_divrem(a: &QPoly, b: &QPoly) -> (QPoly, QPoly) {
    let b = trim(b.clone());
    let bd = degree(&b);
    debug_assert!(bd >= 0, "poly_divrem: division by zero polynomial");
    let lcb = b[bd as usize].clone();

    let mut r = trim(a.clone());
    let ad = degree(&r);
    if ad < bd {
        return (poly_zero(), r);
    }
    let mut q = vec![Rational::from(0); (ad - bd + 1) as usize];

    loop {
        let rd = degree(&r);
        if rd < bd {
            break;
        }
        let shift = (rd - bd) as usize;
        let factor = r[rd as usize].clone() / lcb.clone();
        q[shift] += factor.clone();
        for (i, bc) in b.iter().enumerate() {
            r[shift + i] -= factor.clone() * bc.clone();
        }
        r = trim(r);
        if r.is_empty() {
            break;
        }
    }
    (trim(q), trim(r))
}

/// Make a polynomial monic (leading coefficient 1).  The zero polynomial is
/// returned unchanged.
fn poly_monic(p: &QPoly) -> QPoly {
    let p = trim(p.clone());
    let d = degree(&p);
    if d < 0 {
        return p;
    }
    let lc = p[d as usize].clone();
    poly_scale(&p, &(Rational::from(1) / lc))
}

/// Monic GCD of `a` and `b` over ℚ (Euclidean algorithm).
fn poly_gcd(a: &QPoly, b: &QPoly) -> QPoly {
    let mut a = trim(a.clone());
    let mut b = trim(b.clone());
    while !b.is_empty() {
        let (_, r) = poly_divrem(&a, &b);
        a = b;
        b = r;
    }
    poly_monic(&a)
}

/// Exact division `a / b` (panics in debug if the remainder is nonzero).
fn poly_div_exact(a: &QPoly, b: &QPoly) -> QPoly {
    let (q, r) = poly_divrem(a, b);
    debug_assert!(trim(r).is_empty(), "poly_div_exact: nonzero remainder");
    q
}

/// `p^n` for `n ≥ 0`.
fn poly_pow(p: &QPoly, n: u32) -> QPoly {
    let mut acc = poly_one();
    for _ in 0..n {
        acc = poly_mul(&acc, p);
    }
    acc
}

fn polys_equal(a: &QPoly, b: &QPoly) -> bool {
    trim(a.clone()) == trim(b.clone())
}

// ---------------------------------------------------------------------------
// Exact linear system solver over ℚ
// ---------------------------------------------------------------------------

/// Solve `mat · x = rhs` over ℚ by Gauss–Jordan elimination.
///
/// Returns a particular solution (free variables set to 0), or `None` if the
/// system is inconsistent.  `mat` is `rows × cols`.
fn solve_linear_system(
    mut mat: Vec<Vec<Rational>>,
    mut rhs: Vec<Rational>,
    cols: usize,
) -> Option<Vec<Rational>> {
    let rows = mat.len();
    let mut pivot_row_of_col: Vec<Option<usize>> = vec![None; cols];
    let mut row = 0usize;

    for col in 0..cols {
        if row >= rows {
            break;
        }
        // Find a pivot in this column at or below `row`.
        let Some(sel) = (row..rows).find(|&r| mat[r][col] != 0) else {
            continue;
        };
        mat.swap(row, sel);
        rhs.swap(row, sel);

        // Normalise the pivot row.
        let piv = mat[row][col].clone();
        for cell in mat[row][col..cols].iter_mut() {
            *cell /= piv.clone();
        }
        rhs[row] /= piv.clone();

        // Eliminate the column from every other row.
        let pivot_row = mat[row].clone();
        let pivot_rhs = rhs[row].clone();
        for r in 0..rows {
            if r != row && mat[r][col] != 0 {
                let factor = mat[r][col].clone();
                for (cell, pv) in mat[r][col..cols]
                    .iter_mut()
                    .zip(pivot_row[col..cols].iter())
                {
                    *cell -= factor.clone() * pv.clone();
                }
                rhs[r] -= factor.clone() * pivot_rhs.clone();
            }
        }
        pivot_row_of_col[col] = Some(row);
        row += 1;
    }

    // Consistency: an all-zero row in `mat` with nonzero `rhs` has no solution.
    for r in 0..rows {
        if mat[r].iter().all(|v| *v == 0) && rhs[r] != 0 {
            return None;
        }
    }

    let mut x = vec![Rational::from(0); cols];
    for (col, pr) in pivot_row_of_col.iter().enumerate() {
        if let Some(pr) = pr {
            x[col] = rhs[*pr].clone();
        }
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// Rational RDE solver
// ---------------------------------------------------------------------------

/// Solve `v' + f·v = c_num/c_den` for `v ∈ ℚ(x)`.
///
/// `f` is a polynomial (the exp-tower coefficient `k·η'`).  Returns the solution
/// as a reduced `(numerator, denominator)` pair, or `None` if no rational
/// solution exists (which certifies a non-elementary integral in the exp tower).
pub fn solve_rational_rde(f: &QPoly, c_num: &QPoly, c_den: &QPoly) -> Option<(QPoly, QPoly)> {
    let c_num = trim(c_num.clone());
    let c_den = trim(c_den.clone());

    // c = 0 → v = 0.
    if c_num.is_empty() {
        return Some((poly_zero(), poly_one()));
    }
    if c_den.is_empty() {
        return None; // division by zero — malformed input
    }

    // Reduce c = C/B to lowest terms with B monic.
    let g = poly_gcd(&c_num, &c_den);
    let big_c = poly_div_exact(&c_num, &g);
    let b_raw = poly_div_exact(&c_den, &g);
    // Scale so that B is monic, applying the same scale to C.
    let bd = degree(&b_raw);
    let scale = Rational::from(1) / b_raw[bd as usize].clone();
    let big_b = poly_scale(&b_raw, &scale);
    let big_c = poly_scale(&big_c, &scale);

    // Denominator bound for v: E = gcd(B, B').  G = B / E.
    let bprime = poly_deriv(&big_b);
    let e_poly = poly_gcd(&big_b, &bprime);
    let g_poly = poly_div_exact(&big_b, &e_poly);
    let eprime = poly_deriv(&e_poly);

    // Precompute the polynomial multipliers of the linear identity
    //   Σ_j n_j · P_j(x) = C·E,   P_j = G·E·(j x^{j-1}) − G·E'·x^j + G·E·f·x^j.
    let ge = poly_mul(&g_poly, &e_poly); // G·E
    let gep = poly_mul(&g_poly, &eprime); // G·E'
    let gef = poly_mul(&ge, f); // G·E·f
    let target = poly_mul(&big_c, &e_poly); // C·E

    // Degree bound for N (= numerator of v = N/E).
    let deg_b = degree(&big_b);
    let deg_c = degree(&big_c);
    let deg_e = degree(&e_poly).max(0);
    let deg_f = degree(f).max(0);
    let poly_part = (deg_c - deg_b).max(0);
    let dbound = (deg_e + poly_part.max(deg_f) + 2).max(0) as usize;
    let cols = dbound + 1; // unknowns n_0..n_dbound

    // Maximum degree appearing in the identity.
    let max_deg = (degree(&gef) + dbound as i64)
        .max(degree(&ge) + dbound as i64)
        .max(degree(&gep) + dbound as i64)
        .max(degree(&target))
        .max(0) as usize;
    let n_rows = max_deg + 1;

    // Assemble the linear system M·n = target.
    let mut mat = vec![vec![Rational::from(0); cols]; n_rows];
    for (d, row) in mat.iter_mut().enumerate() {
        let d = d as i64;
        for (j, cell) in row.iter_mut().enumerate() {
            let jj = j as i64;
            // [G·E·(j x^{j-1})]_d = j · (G·E)[d-j+1]
            let mut v = Rational::from(jj) * coeff(&ge, d - jj + 1);
            // − [G·E'·x^j]_d = −(G·E')[d-j]
            v -= coeff(&gep, d - jj);
            // + [G·E·f·x^j]_d = (G·E·f)[d-j]
            v += coeff(&gef, d - jj);
            *cell = v;
        }
    }
    let rhs: Vec<Rational> = (0..n_rows).map(|d| coeff(&target, d as i64)).collect();

    let solution = solve_linear_system(mat, rhs, cols)?;
    let n_poly = trim(solution);

    // Verify: (N'E − N E' + f N E)·B == C·E²   (i.e. v'+f v == C/B with v=N/E).
    let np = poly_deriv(&n_poly);
    let lhs = poly_mul(
        &poly_add(
            &poly_sub(&poly_mul(&np, &e_poly), &poly_mul(&n_poly, &eprime)),
            &poly_mul(&poly_mul(f, &n_poly), &e_poly),
        ),
        &big_b,
    );
    let rhs_check = poly_mul(&big_c, &poly_mul(&e_poly, &e_poly));
    if !polys_equal(&lhs, &rhs_check) {
        return None;
    }

    // Reduce v = N/E to lowest terms.
    if n_poly.is_empty() {
        return Some((poly_zero(), poly_one()));
    }
    let gve = poly_gcd(&n_poly, &e_poly);
    let num = poly_div_exact(&n_poly, &gve);
    let den = poly_monic(&poly_div_exact(&e_poly, &gve));
    Some((num, den))
}

// ---------------------------------------------------------------------------
// Conversion: ExprId → rational function (numerator, denominator) over ℚ
// ---------------------------------------------------------------------------

use crate::kernel::{ExprData, ExprId, ExprPool};

/// Parse `expr` as a rational function in `var` over ℚ, returning
/// `(numerator, denominator)` as `QPoly`s, or `None` if it is not a rational
/// function (e.g. contains a transcendental generator or a foreign symbol).
pub fn expr_to_qrational(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(QPoly, QPoly)> {
    if expr == var {
        return Some((vec![Rational::from(0), Rational::from(1)], poly_one()));
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some((vec![Rational::from(n.0.to_i64()?)], poly_one())),
        ExprData::Rational(r) => Some((vec![r.0.clone()], poly_one())),
        ExprData::Add(args) => {
            let mut acc = (poly_zero(), poly_one());
            for a in &args {
                let term = expr_to_qrational(*a, var, pool)?;
                acc = rat_add(&acc, &term);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = (poly_one(), poly_one());
            for a in &args {
                let factor = expr_to_qrational(*a, var, pool)?;
                acc = rat_mul(&acc, &factor);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let n = match pool.get(exp) {
                ExprData::Integer(n) => n.0.to_i64()?,
                _ => return None,
            };
            let (bn, bd) = expr_to_qrational(base, var, pool)?;
            if n >= 0 {
                Some((poly_pow(&bn, n as u32), poly_pow(&bd, n as u32)))
            } else {
                let m = (-n) as u32;
                if trim(bn.clone()).is_empty() {
                    return None; // 1 / 0
                }
                Some((poly_pow(&bd, m), poly_pow(&bn, m)))
            }
        }
        _ => None,
    }
}

fn rat_add(a: &(QPoly, QPoly), b: &(QPoly, QPoly)) -> (QPoly, QPoly) {
    // a.0/a.1 + b.0/b.1 = (a.0·b.1 + b.0·a.1) / (a.1·b.1)
    let num = poly_add(&poly_mul(&a.0, &b.1), &poly_mul(&b.0, &a.1));
    let den = poly_mul(&a.1, &b.1);
    (num, den)
}

fn rat_mul(a: &(QPoly, QPoly), b: &(QPoly, QPoly)) -> (QPoly, QPoly) {
    (poly_mul(&a.0, &b.0), poly_mul(&a.1, &b.1))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }

    // ∫ (x-1)/x² · exp(x) dx = exp(x)/x.
    // RDE: v' + v = (x-1)/x²  →  v = 1/x.
    #[test]
    fn rational_elementary_exp_x() {
        let f = vec![rat(1)]; // f = 1 (η = x, k = 1)
        let c_num = vec![rat(-1), rat(1)]; // x - 1
        let c_den = vec![rat(0), rat(0), rat(1)]; // x²
        let sol = solve_rational_rde(&f, &c_num, &c_den).expect("elementary");
        // v = 1/x  ⇒  num = 1, den = x.
        assert_eq!(trim(sol.0.clone()), vec![rat(1)], "numerator should be 1");
        assert_eq!(trim(sol.1.clone()), vec![rat(0), rat(1)], "denominator x");
    }

    // ∫ x²/(x+1) · exp(x) dx is NON-elementary (leaves an Ei term).
    #[test]
    fn rational_nonelementary_x2_over_x_plus_1() {
        let f = vec![rat(1)];
        let c_num = vec![rat(0), rat(0), rat(1)]; // x²
        let c_den = vec![rat(1), rat(1)]; // x + 1
        assert!(
            solve_rational_rde(&f, &c_num, &c_den).is_none(),
            "x²/(x+1)·exp(x) must be certified non-elementary"
        );
    }

    // ∫ exp(x)/x dx = Ei(x): RDE v' + v = 1/x has no rational solution.
    #[test]
    fn rational_nonelementary_one_over_x() {
        let f = vec![rat(1)];
        let c_num = vec![rat(1)]; // 1
        let c_den = vec![rat(0), rat(1)]; // x
        assert!(solve_rational_rde(&f, &c_num, &c_den).is_none());
    }

    // exp(x)/x² is non-elementary (residual 1/x simple pole → Ei).
    #[test]
    fn rational_nonelementary_one_over_x2() {
        let f = vec![rat(1)];
        let c_num = vec![rat(1)];
        let c_den = vec![rat(0), rat(0), rat(1)]; // x²
        assert!(solve_rational_rde(&f, &c_num, &c_den).is_none());
    }

    // Polynomial RHS still works through the rational solver (E = 1).
    // ∫ x·exp(x²) dx: f = 2x, c = x  →  v = 1/2.
    #[test]
    fn rational_reduces_to_polynomial_case() {
        let f = vec![rat(0), rat(2)]; // 2x
        let c_num = vec![rat(0), rat(1)]; // x
        let c_den = poly_one();
        let sol = solve_rational_rde(&f, &c_num, &c_den).expect("elementary");
        assert_eq!(trim(sol.0), vec![Rational::from((1, 2))]);
        assert_eq!(trim(sol.1), vec![rat(1)]);
    }

    // gcd / divrem sanity.
    #[test]
    fn divrem_gcd_basic() {
        // (x² − 1) = (x + 1)(x − 1) + 0
        let a = vec![rat(-1), rat(0), rat(1)];
        let b = vec![rat(1), rat(1)];
        let (q, r) = poly_divrem(&a, &b);
        assert_eq!(trim(q), vec![rat(-1), rat(1)]); // x − 1
        assert!(trim(r).is_empty());
        // gcd(x²−1, x²−2x+1) = x − 1 (monic)
        let c = vec![rat(1), rat(-2), rat(1)];
        let g = poly_gcd(&a, &c);
        assert_eq!(trim(g), vec![rat(-1), rat(1)]);
    }

    #[test]
    fn qrational_parse() {
        use crate::kernel::{Domain, ExprPool};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // (x - 1)/x²
        let num = pool.add(vec![x, pool.integer(-1_i32)]);
        let den = pool.pow(x, pool.integer(-2_i32));
        let expr = pool.mul(vec![num, den]);
        let (n, d) = expr_to_qrational(expr, x, &pool).expect("parse");
        // Should equal (x-1)/x² up to a common factor.
        // Cross-check: n · x² == d · (x-1).
        let lhs = poly_mul(&n, &vec![rat(0), rat(0), rat(1)]);
        let rhs = poly_mul(&d, &vec![rat(-1), rat(1)]);
        assert!(polys_equal(&lhs, &rhs), "n={n:?} d={d:?}");
    }
}
