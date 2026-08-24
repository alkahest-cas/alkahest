//! A **three-valued decision procedure** for the one Risch differential
//! equation the `∫B√P` route depends on:
//!
//! ```text
//!     v′ + (a′ / 2a)·v = B,        v ∈ ℚ(x),   a squarefree, deg a ≥ 1.
//! ```
//!
//! # Why this exists
//!
//! `(v√a)′ = (v′ + (a′/2a)v)·√a`, so a rational solution `v` is exactly an
//! **algebraic primitive** `∫B√a dx = v·√a`, and *no* rational solution is one
//! of the two premises of the non-elementarity certificate in
//! [`super::genus_zero`] (the other being an empty, certified-complete residue
//! divisor).
//!
//! The general solver `solve_rational_rde_generalized` answers this with
//! `Option`, and its `None` conflates **"no solution exists"** with **"my
//! denominator/degree bound was too weak"**.  A decline is not a disproof, so
//! `None` cannot license a certificate.  This module answers the same question
//! with a verdict that distinguishes the two, by deriving a denominator bound
//! that is *exact* for this particular equation rather than generic:
//!
//! * **Off the branch locus** (`a(α) ≠ 0`): if `v` has a pole of order `k ≥ 1`
//!   at `α` then `v′` has order `k+1` there while `(a′/2a)v` has only order
//!   `k`, so the left side has a pole of order exactly `k+1`.
//! * **On the branch locus** (`a(α) = 0`, simple since `a` is squarefree):
//!   `a′/2a` has a simple pole of residue `½`, so the two terms both reach
//!   order `k+1` with combined leading coefficient `c·(½ − k) ≠ 0` for integer
//!   `k ≥ 1` — again order exactly `k+1`.
//!
//! Either way a pole of `v` of order `k` forces a pole of `B` of order `k+1`,
//! so `den(v)` divides `gcd(den B, (den B)′)`: the denominator bound is
//! **complete**, not heuristic.  With `D` fixed, `v = N/D` turns the equation
//! into the polynomial identity
//!
//! ```text
//!     2a·(N′D − N D′) + a′·N·D  =  2a·D²·B,
//! ```
//!
//! whose left side has degree exactly `m + d + δ − 1` (`m = deg a`,
//! `d = deg N`, `δ = deg D`) unless `d = δ − m/2`; that pins `deg N` too.  What
//! is left is one exact linear system over `ℚ`.  Inconsistent ⇒ **proved** no
//! rational solution.
//!
//! # Honest limitations
//!
//! [`SqrtRde::Undecided`] is returned — never `NoRationalSolution` — when `a`
//! is not squarefree or `deg a < 1`, since the residue-`½` step above is what
//! rules out the cancelling order `k = μ/2` at a root of multiplicity `μ`.

use rug::Rational;

use super::super::risch::poly_rde::{
    degree, poly_add, poly_deriv, poly_mul, poly_scale, trim, QPoly,
};
use super::super::risch::rational_rde::{poly_divrem, poly_gcd, poly_monic, poly_sub};

/// The verdict of [`decide`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum SqrtRde {
    /// `v = num/den` solves the equation (checked by substitution).
    Solved(QPoly, QPoly),
    /// **Proved**: no `v ∈ ℚ(x)` solves it, so `∫B√a dx` has no algebraic
    /// primitive.  This is the only verdict that may support a certificate.
    NoRationalSolution,
    /// Out of the decidable scope.  Says nothing either way.
    Undecided,
}

/// Decide `v′ + (a′/2a)·v = B` over `ℚ(x)` for `B = b_num/b_den`.
pub(super) fn decide(a: &QPoly, b_num: &QPoly, b_den: &QPoly) -> SqrtRde {
    let a = trim(a.clone());
    let m = degree(&a);
    if m < 1 {
        return SqrtRde::Undecided;
    }
    let a_prime = poly_deriv(&a);
    if degree(&poly_gcd(&a, &a_prime)) > 0 {
        return SqrtRde::Undecided; // multiple root: the residue-½ step fails
    }
    let (bn, bd) = match reduce(b_num, b_den) {
        Some(r) => r,
        None => return SqrtRde::Undecided, // zero or malformed denominator
    };
    if bn.is_empty() {
        return SqrtRde::Solved(vec![], vec![Rational::from(1)]); // v ≡ 0
    }

    // Complete denominator bound: den(v) | gcd(bd, bd′).
    let d_poly = {
        let g = poly_gcd(&bd, &poly_deriv(&bd));
        if degree(&g) < 0 {
            vec![Rational::from(1)]
        } else {
            g
        }
    };
    let delta = degree(&d_poly).max(0);

    // RHS = 2a·D²·B as a polynomial; if `bd` does not divide it, the identity
    // is unsatisfiable for *any* numerator, i.e. no rational solution exists.
    let two_a = poly_scale(&a, &Rational::from(2));
    let d_sq = poly_mul(&d_poly, &d_poly);
    let rhs_num = poly_mul(&poly_mul(&two_a, &d_sq), &bn);
    let (rhs, rem) = poly_divrem(&rhs_num, &bd);
    if !trim(rem).is_empty() {
        return SqrtRde::NoRationalSolution;
    }
    let rhs = trim(rhs);

    // Degree bound for N (see the module header).
    let deg_rhs = degree(&rhs);
    let d_max = (deg_rhs - m - delta + 1).max(delta).max(0) as usize;
    if d_max > 512 {
        return SqrtRde::Undecided; // refuse an unreasonable linear system
    }

    // Columns: L(xʲ) = 2a(j·x^{j−1}·D − xʲ·D′) + a′·xʲ·D.
    let d_prime = poly_deriv(&d_poly);
    let mut cols: Vec<QPoly> = Vec::with_capacity(d_max + 1);
    for j in 0..=d_max {
        let xj = monomial(j);
        let dxj = if j == 0 {
            vec![]
        } else {
            let mut p = monomial(j - 1);
            p[j - 1] = Rational::from(j as u32);
            p
        };
        let wronskian = poly_sub(&poly_mul(&dxj, &d_poly), &poly_mul(&xj, &d_prime));
        let col = trim(poly_add(
            &poly_mul(&two_a, &wronskian),
            &poly_mul(&poly_mul(&a_prime, &xj), &d_poly),
        ));
        cols.push(col);
    }

    let rows = cols
        .iter()
        .map(degree)
        .chain(std::iter::once(deg_rhs))
        .max()
        .unwrap_or(-1);
    if rows < 0 {
        return SqrtRde::NoRationalSolution; // RHS ≠ 0 was established above
    }
    let rows = rows as usize + 1;
    let mut mat: Vec<Vec<Rational>> = (0..rows)
        .map(|i| {
            let mut row: Vec<Rational> = cols
                .iter()
                .map(|c| c.get(i).cloned().unwrap_or_else(|| Rational::from(0)))
                .collect();
            row.push(rhs.get(i).cloned().unwrap_or_else(|| Rational::from(0)));
            row
        })
        .collect();

    let Some(sol) = solve_exact(&mut mat, d_max + 1) else {
        return SqrtRde::NoRationalSolution;
    };
    let n_poly = trim(sol);

    // Substitute back: the certificate must not rest on the linear algebra alone.
    let lhs = trim(poly_add(
        &poly_mul(
            &two_a,
            &poly_sub(
                &poly_mul(&poly_deriv(&n_poly), &d_poly),
                &poly_mul(&n_poly, &d_prime),
            ),
        ),
        &poly_mul(&poly_mul(&a_prime, &n_poly), &d_poly),
    ));
    if lhs != rhs {
        return SqrtRde::Undecided;
    }
    SqrtRde::Solved(n_poly, d_poly)
}

/// `xʲ`.
fn monomial(j: usize) -> QPoly {
    let mut p = vec![Rational::from(0); j + 1];
    p[j] = Rational::from(1);
    p
}

/// `b_num/b_den` in lowest terms with a monic denominator; `None` if `b_den`
/// is the zero polynomial.
fn reduce(b_num: &QPoly, b_den: &QPoly) -> Option<(QPoly, QPoly)> {
    let num = trim(b_num.clone());
    let den = trim(b_den.clone());
    if den.is_empty() {
        return None;
    }
    if num.is_empty() {
        return Some((vec![], vec![Rational::from(1)]));
    }
    let g = poly_gcd(&num, &den);
    let (num, _) = poly_divrem(&num, &g);
    let (den, _) = poly_divrem(&den, &g);
    let lc = den[degree(&den) as usize].clone();
    Some((
        poly_scale(&num, &(Rational::from(1) / lc)),
        poly_monic(&den),
    ))
}

/// Exact Gaussian elimination on an augmented matrix with `n` unknowns.
/// `None` when the system is inconsistent.  Free variables are set to zero.
fn solve_exact(mat: &mut [Vec<Rational>], n: usize) -> Option<QPoly> {
    let rows = mat.len();
    let mut pivot_of_col = vec![usize::MAX; n];
    let mut r = 0;
    for c in 0..n {
        let Some(p) = (r..rows).find(|&i| mat[i][c] != 0) else {
            continue;
        };
        mat.swap(r, p);
        let inv = Rational::from(1) / mat[r][c].clone();
        for v in mat[r].iter_mut() {
            *v *= &inv;
        }
        for i in 0..rows {
            if i != r && mat[i][c] != 0 {
                let f = mat[i][c].clone();
                let pivot_row = mat[r].clone();
                for (dst, src) in mat[i].iter_mut().zip(&pivot_row).skip(c) {
                    *dst -= f.clone() * src;
                }
            }
        }
        pivot_of_col[c] = r;
        r += 1;
        if r == rows {
            break;
        }
    }
    // Inconsistency: a row `0 … 0 | nonzero`.
    for row in mat.iter() {
        if row[..n].iter().all(|v| *v == 0) && row[n] != 0 {
            return None;
        }
    }
    let mut sol = vec![Rational::from(0); n];
    for (c, &p) in pivot_of_col.iter().enumerate() {
        if p != usize::MAX {
            sol[c] = mat[p][n].clone();
        }
    }
    Some(sol)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }

    /// `∫ (P′/2)·dx/√P = √P`: `B = a′/2a` has the solution `v = 1`.
    #[test]
    fn solves_the_obvious_algebraic_primitive() {
        let a = qp(&[1, 0, 0, 0, 0, 1]); // x⁵ + 1
        let ap = poly_deriv(&a);
        let two_a = poly_scale(&a, &Rational::from(2));
        match decide(&a, &ap, &two_a) {
            SqrtRde::Solved(n, d) => {
                // v = 1 (up to the reduction's normalisation).
                assert_eq!(degree(&n), 0);
                assert_eq!(degree(&d), 0);
                assert_eq!(
                    n[0].clone() / d[0].clone(),
                    Rational::from(1),
                    "expected v = 1"
                );
            }
            other => panic!("expected Solved, got {other:?}"),
        }
    }

    /// `∫5x⁴√(x⁵+1) dx = ⅔(x⁵+1)^{3/2}`: `B = 5x⁴`, `v = ⅔(x⁵+1)`.
    #[test]
    fn solves_polynomial_weight() {
        let a = qp(&[1, 0, 0, 0, 0, 1]);
        let b = qp(&[0, 0, 0, 0, 5]);
        match decide(&a, &b, &qp(&[1])) {
            SqrtRde::Solved(n, d) => {
                assert_eq!(degree(&d), 0);
                let v: Vec<Rational> = n.iter().map(|c| c.clone() / d[0].clone()).collect();
                assert_eq!(
                    v,
                    vec![
                        Rational::from((2, 3)),
                        Rational::from(0),
                        Rational::from(0),
                        Rational::from(0),
                        Rational::from(0),
                        Rational::from((2, 3))
                    ]
                );
            }
            other => panic!("expected Solved, got {other:?}"),
        }
    }

    /// `∫dx/√(x⁵+1)`: `B = 1/(x⁵+1)`.  There is **no** algebraic primitive —
    /// and this is a proof, not a decline, so the certificate downstream is
    /// entitled to rest on it.
    #[test]
    fn proves_no_algebraic_primitive_for_the_quintic_first_kind() {
        let a = qp(&[1, 0, 0, 0, 0, 1]);
        assert_eq!(
            decide(&a, &qp(&[1]), &a.clone()),
            SqrtRde::NoRationalSolution
        );
    }

    /// `∫x dx/√(1−x⁴)` also has no algebraic primitive — the integral is
    /// `½asin(x²)`, a *logarithmic* part, which is why premise B alone never
    /// certifies anything.
    #[test]
    fn proves_no_algebraic_primitive_for_the_elementary_pullback() {
        let a = qp(&[1, 0, 0, 0, -1]);
        assert_eq!(
            decide(&a, &qp(&[0, 1]), &a.clone()),
            SqrtRde::NoRationalSolution
        );
    }

    /// `∫x dx/√(1−x⁶)`: no algebraic primitive either.
    #[test]
    fn proves_no_algebraic_primitive_for_the_genus_two_first_kind() {
        let a = qp(&[1, 0, 0, 0, 0, 0, -1]);
        assert_eq!(
            decide(&a, &qp(&[0, 1]), &a.clone()),
            SqrtRde::NoRationalSolution
        );
    }

    /// A non-squarefree radicand is refused rather than pronounced upon.
    #[test]
    fn refuses_non_squarefree_radicand() {
        let a = poly_mul(&qp(&[1, 1]), &poly_mul(&qp(&[1, 1]), &qp(&[0, 1])));
        assert_eq!(decide(&a, &qp(&[1]), &qp(&[1])), SqrtRde::Undecided);
    }

    /// `B = 0` is solved by `v = 0`.
    #[test]
    fn zero_weight_is_solved() {
        let a = qp(&[1, 0, 0, 1]);
        assert!(matches!(
            decide(&a, &qp(&[]), &qp(&[1])),
            SqrtRde::Solved(_, _)
        ));
    }

    /// A solution with a genuine denominator: `a = x³+1`, `v = 1/x`, so
    /// `B = v′ + (a′/2a)v = −1/x² + 3x/(2(x³+1))`.
    #[test]
    fn solves_with_a_denominator() {
        let a = qp(&[1, 0, 0, 1]);
        // B = (−2(x³+1) + 3x³) / (2x²(x³+1)) = (x³ − 2)/(2x²(x³+1))
        let bn = qp(&[-2, 0, 0, 1]);
        let bd = poly_scale(&poly_mul(&qp(&[0, 0, 1]), &a), &Rational::from(2));
        match decide(&a, &bn, &bd) {
            SqrtRde::Solved(n, d) => {
                // v = N/D must equal 1/x.
                assert_eq!(trim(poly_mul(&n, &qp(&[0, 1]))), trim(d));
            }
            other => panic!("expected Solved, got {other:?}"),
        }
    }
}
