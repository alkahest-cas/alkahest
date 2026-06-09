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

use super::number_field::{KElem, KPoly, NumberField};
use super::poly_rde::{
    degree, poly_add, poly_deriv, poly_mul, poly_one, poly_scale, poly_zero, trim, QPoly,
};

// ---------------------------------------------------------------------------
// Polynomial arithmetic over ℚ not already provided by `poly_rde`
// ---------------------------------------------------------------------------

/// Subtract `b` from `a`.
pub fn poly_sub(a: &QPoly, b: &QPoly) -> QPoly {
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
pub fn poly_divrem(a: &QPoly, b: &QPoly) -> (QPoly, QPoly) {
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
pub fn poly_monic(p: &QPoly) -> QPoly {
    let p = trim(p.clone());
    let d = degree(&p);
    if d < 0 {
        return p;
    }
    let lc = p[d as usize].clone();
    poly_scale(&p, &(Rational::from(1) / lc))
}

/// Monic GCD of `a` and `b` over ℚ (Euclidean algorithm).
pub fn poly_gcd(a: &QPoly, b: &QPoly) -> QPoly {
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
pub fn poly_div_exact(a: &QPoly, b: &QPoly) -> QPoly {
    let (q, r) = poly_divrem(a, b);
    debug_assert!(trim(r).is_empty(), "poly_div_exact: nonzero remainder");
    q
}

/// `p^n` for `n ≥ 0`.
pub fn poly_pow(p: &QPoly, n: u32) -> QPoly {
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

/// The canonical Bronstein §6.5 base-case bound on the degree of the numerator
/// `N` of a rational solution `v = N/E` of `v' + f·v = c`, where `c = C/B`
/// (reduced, `B` monic) and `E = gcd(B, B')`.
///
/// Concretely `dbound = deg E + max(deg C − deg B, deg f) + 2` (clamped at 0).
/// This is the single source of truth for the base bound; the polymorphic
/// [`DifferentialField::rde_degree_bound`](super::diff_field::DifferentialField::rde_degree_bound)
/// for `ℚ(x)` mirrors it so the ansatz solvers can use it as a search ceiling.
pub(crate) fn numerator_degree_bound(deg_b: i64, deg_c: i64, deg_e: i64, deg_f: i64) -> usize {
    let poly_part = (deg_c - deg_b).max(0);
    (deg_e.max(0) + poly_part.max(deg_f.max(0)) + 2).max(0) as usize
}

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

    // Degree bound for N (= numerator of v = N/E).  See [`numerator_degree_bound`].
    let deg_b = degree(&big_b);
    let deg_c = degree(&big_c);
    let deg_e = degree(&e_poly).max(0);
    let deg_f = degree(f).max(0);
    let dbound = numerator_degree_bound(deg_b, deg_c, deg_e, deg_f);
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
// Generalized rational RDE: f ∈ ℚ(x)  (Risch Gap F — rational exponents)
// ---------------------------------------------------------------------------

/// Solve `v' + f·v = c_num/c_den` for `v ∈ ℚ(x)` where `f = f_num/f_den` is
/// a **rational** function (not necessarily a polynomial).
///
/// When `f_den` is a constant this delegates to [`solve_rational_rde`].
/// Otherwise it uses the generalized identity
/// ```text
///   B_f · G · (N'·E − N·E') + A · G · E · N = B_f · C · E
/// ```
/// where `f = A/B_f` (lowest terms, `B_f` monic), `c = C/D` (lowest terms,
/// `D` monic), `E = gcd(D, D')`, `G = D/E`.  Substituting `v = N/E` and
/// clearing denominators yields this polynomial identity, which is linear in
/// the coefficients of `N`.  The degree bound for `N` is
/// `deg(B_f) + deg(E) + deg(C) + 2` (generous).
///
/// Returns the reduced `(numerator, denominator)` of `v`, or `None` when no
/// rational solution exists (certifying a non-elementary integral).
///
/// References: Bronstein (2005) §5.4, §6.1.
pub fn solve_rational_rde_generalized(
    f_num: &QPoly,
    f_den: &QPoly,
    c_num: &QPoly,
    c_den: &QPoly,
) -> Option<(QPoly, QPoly)> {
    let f_den_t = trim(f_den.clone());
    let f_num_t = trim(f_num.clone());

    // Degenerate: f_den = 0 is undefined input.
    if f_den_t.is_empty() {
        return None;
    }

    // When f_den is a constant, f is a polynomial: scale and use the fast path.
    if degree(&f_den_t) == 0 {
        let scale = Rational::from(1) / f_den_t[0].clone();
        let f_poly = poly_scale(&f_num_t, &scale);
        return solve_rational_rde(&f_poly, c_num, c_den);
    }

    // Reduce f = A / B_f (lowest terms, B_f monic).
    let gf = poly_gcd(&f_num_t, &f_den_t);
    let a_raw = poly_div_exact(&f_num_t, &gf);
    let bf_raw = poly_div_exact(&f_den_t, &gf);
    let bf_d = degree(&bf_raw);
    debug_assert!(bf_d > 0, "f_den should have positive degree here");
    let bf_lc_inv = Rational::from(1) / bf_raw[bf_d as usize].clone();
    let big_a = poly_scale(&a_raw, &bf_lc_inv);
    let big_bf = poly_scale(&bf_raw, &bf_lc_inv);

    // Reduce c = C / D (lowest terms, D monic).
    let c_num_t = trim(c_num.clone());
    let c_den_t = trim(c_den.clone());
    if c_num_t.is_empty() {
        return Some((poly_zero(), poly_one()));
    }
    if c_den_t.is_empty() {
        return None; // c_den = 0: malformed
    }
    let gc = poly_gcd(&c_num_t, &c_den_t);
    let big_c_raw = poly_div_exact(&c_num_t, &gc);
    let d_raw = poly_div_exact(&c_den_t, &gc);
    let d_d = degree(&d_raw);
    let d_lc_inv = if d_d >= 0 {
        Rational::from(1) / d_raw[d_d as usize].clone()
    } else {
        Rational::from(1)
    };
    let big_d = poly_scale(&d_raw, &d_lc_inv);
    let big_c = poly_scale(&big_c_raw, &d_lc_inv);

    // Denominator bound: E = gcd(D, D'),  G = D / E.
    let dprime = poly_deriv(&big_d);
    let e_poly = poly_gcd(&big_d, &dprime);
    let g_poly = poly_div_exact(&big_d, &e_poly);
    let eprime = poly_deriv(&e_poly);

    // Precompute the polynomial combinations that appear in the identity
    //   B_f·G·(N'·E − N·E') + A·G·E·N = B_f·C·E.
    let ge = poly_mul(&g_poly, &e_poly); // G · E
    let bfg = poly_mul(&big_bf, &g_poly); // B_f · G
    let bfge = poly_mul(&bfg, &e_poly); // B_f · G · E  (for the N' term)
    let bfgep = poly_mul(&bfg, &eprime); // B_f · G · E' (for the −N term)
    let age = poly_mul(&big_a, &ge); // A · G · E      (for the +N term)
    let target = poly_mul(&poly_mul(&big_bf, &big_c), &e_poly); // B_f · C · E

    // Degree bound for N (generous: accounts for cancellation in leading terms).
    let deg_bf = degree(&big_bf).max(0) as usize;
    let deg_e = degree(&e_poly).max(0) as usize;
    let deg_c = degree(&big_c).max(0) as usize;
    let deg_target = degree(&target).max(0) as usize;
    let dbound = (deg_bf + deg_e + deg_c + 2).max(deg_target + 1);
    let cols = dbound + 1;

    // Maximum degree of any term in the identity.
    let max_deg = (degree(&bfge) + dbound as i64)
        .max(degree(&bfgep) + dbound as i64)
        .max(degree(&age) + dbound as i64)
        .max(degree(&target))
        .max(0) as usize;
    let n_rows = max_deg + 1;

    // Assemble the linear system M · n = rhs, one equation per degree.
    let mut mat = vec![vec![Rational::from(0); cols]; n_rows];
    for (d, row) in mat.iter_mut().enumerate() {
        let d = d as i64;
        for (j, cell) in row.iter_mut().enumerate() {
            let jj = j as i64;
            // [B_f·G·E · N']_d  = j · (B_f·G·E)[d−j+1]
            let mut v = Rational::from(jj) * coeff(&bfge, d - jj + 1);
            // −[B_f·G·E' · N]_d = −(B_f·G·E')[d−j]
            v -= coeff(&bfgep, d - jj);
            // +[A·G·E · N]_d    = (A·G·E)[d−j]
            v += coeff(&age, d - jj);
            *cell = v;
        }
    }
    let rhs: Vec<Rational> = (0..n_rows).map(|d| coeff(&target, d as i64)).collect();

    let solution = solve_linear_system(mat, rhs, cols)?;
    let n_poly = trim(solution);

    // Verify: B_f·G·(N'·E − N·E') + A·G·E·N == B_f·C·E.
    let np = poly_deriv(&n_poly);
    let lhs = poly_add(
        &poly_mul(
            &bfg,
            &poly_sub(&poly_mul(&np, &e_poly), &poly_mul(&n_poly, &eprime)),
        ),
        &poly_mul(&age, &n_poly),
    );
    if !polys_equal(&lhs, &target) {
        return None;
    }

    // Reduce v = N / E to lowest terms.
    if n_poly.is_empty() {
        return Some((poly_zero(), poly_one()));
    }
    let gve = poly_gcd(&n_poly, &e_poly);
    let num = poly_div_exact(&n_poly, &gve);
    let den = poly_monic(&poly_div_exact(&e_poly, &gve));
    Some((num, den))
}

// ---------------------------------------------------------------------------
// Rational RDE over a number field K = ℚ(α)  (Risch Gap E, rational case)
// ---------------------------------------------------------------------------

/// Solve `v' + f·v = c_num/c_den` for `v ∈ K(x)`, `K = ℚ(α)`.
///
/// This is the number-field analogue of [`solve_rational_rde`]: identical
/// algorithm — denominator bound `E = gcd(B, B')`, ansatz `v = N/E`, the linear
/// identity `Σ_j n_j·P_j = C·E`, and the final substitution check — with every
/// coefficient operation routed through `field` instead of ℚ.  `f`, `c_num`,
/// `c_den` are `K`-polynomials in `x`.
///
/// In the exp tower `f = k·η'` is a polynomial (no poles), so the `E = gcd(B,B')`
/// bound is exact and an inconsistent/over-determined system correctly certifies
/// a non-elementary integral over `K` (the residual simple poles are the Ei/Li
/// part the exp tower cannot express).
pub fn solve_rational_rde_k(
    field: &NumberField,
    f: &KPoly,
    c_num: &KPoly,
    c_den: &KPoly,
) -> Option<(KPoly, KPoly)> {
    let c_num = NumberField::kpoly_trim(c_num.clone());
    let c_den = NumberField::kpoly_trim(c_den.clone());
    let one: KPoly = vec![field.from_int(1)];

    // c = 0 → v = 0.
    if c_num.is_empty() {
        return Some((Vec::new(), one));
    }
    if c_den.is_empty() {
        return None; // division by zero — malformed input
    }

    // Reduce c = C/B to lowest terms with B monic.
    let g = field.kpoly_gcd(&c_num, &c_den)?;
    let big_c = field.kpoly_div_exact(&c_num, &g)?;
    let b_raw = field.kpoly_div_exact(&c_den, &g)?;
    let bd = NumberField::kdeg(&b_raw);
    let lead_inv = field.inv(&b_raw[bd as usize])?;
    let big_b = field.kpoly_scale(&b_raw, &lead_inv);
    let big_c = field.kpoly_scale(&big_c, &lead_inv);

    // Denominator bound for v: E = gcd(B, B'). G = B / E.
    let bprime = field.kpoly_deriv(&big_b);
    let e_poly = field.kpoly_gcd(&big_b, &bprime)?;
    let g_poly = field.kpoly_div_exact(&big_b, &e_poly)?;
    let eprime = field.kpoly_deriv(&e_poly);

    // Polynomial multipliers of the identity  Σ_j n_j·P_j = C·E,
    //   P_j = G·E·(j x^{j-1}) − G·E'·x^j + G·E·f·x^j.
    let ge = field.kpoly_mul(&g_poly, &e_poly);
    let gep = field.kpoly_mul(&g_poly, &eprime);
    let gef = field.kpoly_mul(&ge, f);
    let target = field.kpoly_mul(&big_c, &e_poly);

    // Degree bound for N (= numerator of v = N/E).
    let deg_b = NumberField::kdeg(&big_b);
    let deg_c = NumberField::kdeg(&big_c);
    let deg_e = NumberField::kdeg(&e_poly).max(0);
    let deg_f = NumberField::kdeg(f).max(0);
    let poly_part = (deg_c - deg_b).max(0);
    let dbound = (deg_e + poly_part.max(deg_f) + 2).max(0) as usize;
    let cols = dbound + 1;

    let max_deg = (NumberField::kdeg(&gef) + dbound as i64)
        .max(NumberField::kdeg(&ge) + dbound as i64)
        .max(NumberField::kdeg(&gep) + dbound as i64)
        .max(NumberField::kdeg(&target))
        .max(0) as usize;
    let n_rows = max_deg + 1;

    // Assemble M·n = target over K.
    let mut mat = vec![vec![NumberField::k_zero(); cols]; n_rows];
    for (d, row) in mat.iter_mut().enumerate() {
        let d = d as i64;
        for (j, cell) in row.iter_mut().enumerate() {
            let jj = j as i64;
            // [G·E·(j x^{j-1})]_d = j · (G·E)[d-j+1]
            let mut v = field.mul(&field.from_int(jj), &NumberField::kcoeff(&ge, d - jj + 1));
            // − [G·E'·x^j]_d = −(G·E')[d-j]
            v = field.sub(&v, &NumberField::kcoeff(&gep, d - jj));
            // + [G·E·f·x^j]_d = (G·E·f)[d-j]
            v = field.add(&v, &NumberField::kcoeff(&gef, d - jj));
            *cell = v;
        }
    }
    let rhs: Vec<KElem> = (0..n_rows)
        .map(|d| NumberField::kcoeff(&target, d as i64))
        .collect();

    let solution = solve_linear_system_k(field, mat, rhs, cols)?;
    let n_poly = NumberField::kpoly_trim(solution);

    // Verify (N'E − N E' + f N E)·B == C·E².
    let np = field.kpoly_deriv(&n_poly);
    let lhs = field.kpoly_mul(
        &field.kpoly_add(
            &field.kpoly_sub(
                &field.kpoly_mul(&np, &e_poly),
                &field.kpoly_mul(&n_poly, &eprime),
            ),
            &field.kpoly_mul(&field.kpoly_mul(f, &n_poly), &e_poly),
        ),
        &big_b,
    );
    let rhs_check = field.kpoly_mul(&big_c, &field.kpoly_mul(&e_poly, &e_poly));
    if !NumberField::kpoly_eq(&lhs, &rhs_check) {
        return None;
    }

    // Reduce v = N/E to lowest terms.
    if n_poly.is_empty() {
        return Some((Vec::new(), one));
    }
    let gve = field.kpoly_gcd(&n_poly, &e_poly)?;
    let num = field.kpoly_div_exact(&n_poly, &gve)?;
    let den = field.kpoly_monic(&field.kpoly_div_exact(&e_poly, &gve)?)?;
    Some((num, den))
}

/// Solve `mat · x = rhs` over a number field `K` by Gauss–Jordan elimination.
/// Returns a particular solution (free variables 0), or `None` if inconsistent.
fn solve_linear_system_k(
    field: &NumberField,
    mut mat: Vec<Vec<KElem>>,
    mut rhs: Vec<KElem>,
    cols: usize,
) -> Option<Vec<KElem>> {
    let rows = mat.len();
    let mut pivot_row_of_col: Vec<Option<usize>> = vec![None; cols];
    let mut row = 0usize;

    for col in 0..cols {
        if row >= rows {
            break;
        }
        let Some(sel) = (row..rows).find(|&r| !NumberField::is_zero(&mat[r][col])) else {
            continue;
        };
        mat.swap(row, sel);
        rhs.swap(row, sel);

        // Normalise the pivot row.
        let piv_inv = field.inv(&mat[row][col])?;
        for cell in mat[row][col..cols].iter_mut() {
            *cell = field.mul(cell, &piv_inv);
        }
        rhs[row] = field.mul(&rhs[row], &piv_inv);

        // Eliminate the column from every other row.
        let pivot_row = mat[row].clone();
        let pivot_rhs = rhs[row].clone();
        for r in 0..rows {
            if r != row && !NumberField::is_zero(&mat[r][col]) {
                let factor = mat[r][col].clone();
                for (cell, pv) in mat[r][col..cols]
                    .iter_mut()
                    .zip(pivot_row[col..cols].iter())
                {
                    *cell = field.sub(cell, &field.mul(&factor, pv));
                }
                rhs[r] = field.sub(&rhs[r], &field.mul(&factor, &pivot_rhs));
            }
        }
        pivot_row_of_col[col] = Some(row);
        row += 1;
    }

    // Consistency: an all-zero row with nonzero rhs has no solution.
    for r in 0..rows {
        if mat[r].iter().all(NumberField::is_zero) && !NumberField::is_zero(&rhs[r]) {
            return None;
        }
    }

    let mut x = vec![NumberField::k_zero(); cols];
    for (col, pr) in pivot_row_of_col.iter().enumerate() {
        if let Some(pr) = pr {
            x[col] = rhs[*pr].clone();
        }
    }
    Some(x)
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

    // -----------------------------------------------------------------------
    // Rational RDE over a number field K = ℚ(√d)  (Gap E, rational case)
    // -----------------------------------------------------------------------

    /// ℚ(√2) = ℚ[t]/(t²−2).
    fn field_sqrt2() -> NumberField {
        NumberField::new(vec![rat(-2), rat(0), rat(1)])
    }

    /// A K-constant from a rational.
    fn kc(field: &NumberField, n: i64) -> KElem {
        field.from_int(n)
    }

    // ∫ (x − √2 − 1)/(x − √2)² · exp(x) dx = exp(x)/(x − √2).
    // RDE: v' + v = (x−√2−1)/(x−√2)²  →  v = 1/(x − √2).
    #[test]
    fn rational_rde_k_elementary_sqrt2() {
        let field = field_sqrt2();
        let f: KPoly = vec![kc(&field, 1)]; // f = 1 (η = x, k = 1)
        let sqrt2 = vec![rat(0), rat(1)]; // √2 as a K-element
                                          // c_num = x − √2 − 1: x^0 = −√2 − 1, x^1 = 1.
        let c0 = field.sub(&field.neg(&sqrt2), &kc(&field, 1));
        let c_num: KPoly = vec![c0, kc(&field, 1)];
        // c_den = (x − √2)² = x² − 2√2·x + 2.
        let base: KPoly = vec![field.neg(&sqrt2), kc(&field, 1)]; // x − √2
        let c_den = field.kpoly_mul(&base, &base);

        let (vn, vd) = solve_rational_rde_k(&field, &f, &c_num, &c_den).expect("elementary");
        // v = 1/(x − √2): num = 1, den = x − √2 (monic).
        assert_eq!(NumberField::kdeg(&vn), 0);
        assert_eq!(trim(vn[0].clone()), vec![rat(1)]);
        assert!(NumberField::kpoly_eq(&vd, &base));
    }

    // ∫ x²/(x − √2) · exp(x) dx is non-elementary (Ei term: simple pole, residue 2).
    #[test]
    fn rational_rde_k_nonelementary_sqrt2() {
        let field = field_sqrt2();
        let f: KPoly = vec![kc(&field, 1)];
        let sqrt2 = vec![rat(0), rat(1)];
        let c_num: KPoly = vec![NumberField::k_zero(), NumberField::k_zero(), kc(&field, 1)]; // x²
        let c_den: KPoly = vec![field.neg(&sqrt2), kc(&field, 1)]; // x − √2
        assert!(solve_rational_rde_k(&field, &f, &c_num, &c_den).is_none());
    }

    // A polynomial RHS still works through the K rational solver (E = 1).
    // ∫ x·exp(x²) dx: f = 2x, c = x  →  v = 1/2 (a K-constant).
    #[test]
    fn rational_rde_k_reduces_to_polynomial() {
        let field = field_sqrt2();
        let f: KPoly = vec![NumberField::k_zero(), kc(&field, 2)]; // 2x
        let c_num: KPoly = vec![NumberField::k_zero(), kc(&field, 1)]; // x
        let c_den: KPoly = vec![kc(&field, 1)]; // 1
        let (vn, vd) = solve_rational_rde_k(&field, &f, &c_num, &c_den).expect("elementary");
        assert_eq!(trim(vn[0].clone()), vec![Rational::from((1, 2))]);
        assert_eq!(trim(vd[0].clone()), vec![rat(1)]);
    }

    // -----------------------------------------------------------------------
    // Generalized rational RDE: f ∈ ℚ(x)  (Gap F — rational exponents)
    // -----------------------------------------------------------------------

    // ∫ exp(1/x) dx: RDE v' − (1/x²)·v = 1, no rational solution.
    // f = −1/x² (f_num = −1, f_den = x²), c = 1.
    #[test]
    fn gen_rde_exp_inv_x_nonelementary() {
        let f_num = vec![rat(-1)];
        let f_den = vec![rat(0), rat(0), rat(1)]; // x²
        let c_num = poly_one();
        let c_den = poly_one();
        assert!(
            solve_rational_rde_generalized(&f_num, &f_den, &c_num, &c_den).is_none(),
            "∫ exp(1/x) dx must be certified non-elementary"
        );
    }

    // ∫ (1/x²)·exp(1/x) dx = −exp(1/x).
    // f = −1/x², c = 1/x².  Solution v = −1 = N/E with N = −x, E = x.
    #[test]
    fn gen_rde_inv_x2_exp_inv_x_elementary() {
        let f_num = vec![rat(-1)];
        let f_den = vec![rat(0), rat(0), rat(1)]; // x²
        let c_num = poly_one(); // 1
        let c_den = vec![rat(0), rat(0), rat(1)]; // x²
        let (vn, vd) = solve_rational_rde_generalized(&f_num, &f_den, &c_num, &c_den)
            .expect("∫ (1/x²)·exp(1/x) dx must be elementary");
        // v = −1: num = −1, den = 1.
        assert_eq!(
            trim(vn.clone()),
            vec![rat(-1)],
            "numerator should be −1, got {vn:?}"
        );
        assert_eq!(
            trim(vd.clone()),
            poly_one(),
            "denominator should be 1, got {vd:?}"
        );
    }

    // ∫ (2/x³)·exp(−1/x²) dx = exp(−1/x²).
    // η = −1/x², η' = 2/x³.  f = 2/x³, c = 2/x³.  Solution v = 1.
    #[test]
    fn gen_rde_exp_neg_inv_x2_elementary() {
        let f_num = vec![rat(2)];
        let f_den = vec![rat(0), rat(0), rat(0), rat(1)]; // x³
        let c_num = vec![rat(2)];
        let c_den = vec![rat(0), rat(0), rat(0), rat(1)]; // x³
        let (vn, vd) = solve_rational_rde_generalized(&f_num, &f_den, &c_num, &c_den)
            .expect("∫ (2/x³)·exp(−1/x²) dx must be elementary");
        // v = 1: num = 1, den = 1.
        assert_eq!(
            trim(vn.clone()),
            poly_one(),
            "numerator should be 1, got {vn:?}"
        );
        assert_eq!(
            trim(vd.clone()),
            poly_one(),
            "denominator should be 1, got {vd:?}"
        );
    }

    // Polynomial f falls back to the existing solver correctly.
    // ∫ (x−1)/x²·exp(x) dx = exp(x)/x.  f = 1 (constant den), c = (x−1)/x².
    #[test]
    fn gen_rde_falls_back_to_polynomial_f() {
        let f_num = vec![rat(1)];
        let f_den = poly_one(); // constant denominator → delegate to solve_rational_rde
        let c_num = vec![rat(-1), rat(1)]; // x − 1
        let c_den = vec![rat(0), rat(0), rat(1)]; // x²
        let (vn, vd) = solve_rational_rde_generalized(&f_num, &f_den, &c_num, &c_den)
            .expect("fallback must succeed");
        // v = 1/x.
        assert_eq!(trim(vn), vec![rat(1)]);
        assert_eq!(trim(vd), vec![rat(0), rat(1)]);
    }
}
