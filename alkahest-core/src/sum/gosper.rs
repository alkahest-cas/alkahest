//! Gosper's algorithm on rational hypergeometric ratios (SymPy-compatible normal form).

use super::poly_aux::{compose_affine, gcd_shifted, poly_exact_div, poly_nth_coeff};
use super::ratfunc::{scale_poly, RatFunc};
use crate::matrix::normal_form::RatUniPoly;
use rug::Rational;

/// `(Z·A, B, C)` with `p/q = Z · A(k)·C(k+1) / (B(k)·C(k))` matching SymPy `gosper_normal`.
pub fn gosper_normal_form(
    mut p: RatUniPoly,
    mut q: RatUniPoly,
) -> (RatUniPoly, RatUniPoly, RatUniPoly) {
    if p.is_zero() {
        return (RatUniPoly::zero(), RatUniPoly::one(), RatUniPoly::one());
    }
    if q.is_zero() {
        panic!("gosper_normal_form: denominator is zero");
    }

    let lc_p = p.leading_coeff();
    let lc_q = q.leading_coeff();
    let z_scale = lc_p.clone() / lc_q.clone();

    p = scale_poly(&p, &(Rational::from(1) / lc_p));
    q = scale_poly(&q, &(Rational::from(1) / lc_q));

    let mut a = p;
    let mut b = q;
    let mut c = RatUniPoly::one();

    let bound = (a.degree().max(0) + b.degree().max(0)).max(1) as usize + 32;

    loop {
        let mut found = false;
        for i in 1..=bound {
            let d = gcd_shifted(&a, &b, i as i64);
            if d.is_zero() {
                continue;
            }
            let trivial_unit = d.degree() == 0 && d.coeffs.len() == 1 && d.coeffs[0] == 1;
            if trivial_unit {
                continue;
            }
            let Some(an) = poly_exact_div(&a, &d) else {
                continue;
            };
            let dsmi = compose_affine(&d, &Rational::from(1), &Rational::from(-(i as i64)));
            let Some(bn) = poly_exact_div(&b, &dsmi) else {
                continue;
            };
            a = an;
            b = bn;
            let mut prod = RatUniPoly::one();
            for j in 1..=i {
                prod = prod * compose_affine(&d, &Rational::from(1), &Rational::from(-(j as i64)));
            }
            c = c * prod;
            found = true;
            break;
        }
        if !found {
            break;
        }
    }

    a = scale_poly(&a, &z_scale);
    (a, b, c)
}

fn x_pow(j: usize) -> RatUniPoly {
    let mut coeffs = vec![Rational::from(0); j + 1];
    coeffs[j] = Rational::from(1);
    RatUniPoly { coeffs }.trim()
}

/// Degree candidates for the certificate polynomial `x` (SymPy `gosper_term`).
fn gosper_degree_candidates(a: &RatUniPoly, b_shifted: &RatUniPoly, c: &RatUniPoly) -> Vec<i32> {
    let n = a.degree();
    let m = b_shifted.degree();
    let kdeg = c.degree();
    let mut out = Vec::new();

    if n != m || a.leading_coeff() != b_shifted.leading_coeff() {
        out.push(kdeg - n.max(m));
    } else if n == 0 {
        out.push(kdeg - n + 1);
        out.push(0);
    } else {
        out.push(kdeg - n + 1);
        let lc = a.leading_coeff();
        if lc != Rational::from(0) {
            let pb = poly_nth_coeff(b_shifted, n - 1);
            let pa = poly_nth_coeff(a, n - 1);
            let diff = (pb - pa) / lc.clone();
            let di = {
                let nn = diff.numer().to_f64();
                let dd = diff.denom().to_f64();
                if dd != 0.0 {
                    nn / dd
                } else {
                    f64::NAN
                }
            };
            if di.is_finite() && (di - di.round()).abs() < 1e-9 {
                out.push(di.round() as i32);
            }
        }
    }

    out.into_iter().filter(|&d| d >= 0).collect()
}

fn rational_gaussian_solve(
    mut mat: Vec<Vec<Rational>>,
    mut rhs: Vec<Rational>,
) -> Option<Vec<Rational>> {
    let nrows = mat.len();
    if nrows == 0 {
        return Some(vec![]);
    }
    let ncols = mat[0].len();
    let mut row = 0;
    for col in 0..ncols {
        if row >= nrows {
            break;
        }
        let mut piv = None;
        for r in row..nrows {
            if mat[r][col] != Rational::from(0) {
                piv = Some(r);
                break;
            }
        }
        let pr = piv?;
        mat.swap(row, pr);
        rhs.swap(row, pr);
        let factor = mat[row][col].clone();
        if factor == Rational::from(0) {
            continue;
        }
        let inv = Rational::from(1) / factor.clone();
        for j in col..ncols {
            mat[row][j] *= inv.clone();
        }
        rhs[row] *= inv;
        for r in 0..nrows {
            if r == row {
                continue;
            }
            let v = mat[r][col].clone();
            if v == Rational::from(0) {
                continue;
            }
            for j in col..ncols {
                let t = mat[row][j].clone() * v.clone();
                mat[r][j] -= t;
            }
            let tr = rhs[row].clone() * v;
            rhs[r] -= tr;
        }
        row += 1;
    }

    for r in 0..nrows {
        let all_zero = (0..ncols).all(|j| mat[r][j] == Rational::from(0));
        if all_zero && rhs[r] != Rational::from(0) {
            return None;
        }
    }

    let mut sol = vec![Rational::from(0); ncols];
    for r in (0..nrows).rev() {
        let mut first = None;
        for j in 0..ncols {
            if mat[r][j] != Rational::from(0) {
                first = Some(j);
                break;
            }
        }
        if let Some(j) = first {
            let mut sum = rhs[r].clone();
            for k in (j + 1)..ncols {
                sum -= mat[r][k].clone() * sol[k].clone();
            }
            sol[j] = sum / mat[r][j].clone();
        }
    }
    Some(sol)
}

fn solve_gosper_poly(
    a: &RatUniPoly,
    b_at_k_minus_1: &RatUniPoly,
    c: &RatUniPoly,
    max_degree: usize,
) -> Option<RatUniPoly> {
    for d in 0..=max_degree {
        if let Some(x) = solve_gosper_poly_fixed_degree(a, b_at_k_minus_1, c, d) {
            if !x.is_zero() {
                return Some(x);
            }
        }
    }
    None
}

fn solve_gosper_poly_fixed_degree(
    a: &RatUniPoly,
    b_at_k_minus_1: &RatUniPoly,
    c: &RatUniPoly,
    deg_x: usize,
) -> Option<RatUniPoly> {
    let mut basis_polys: Vec<RatUniPoly> = Vec::with_capacity(deg_x + 1);
    for j in 0..=deg_x {
        let mono_j = x_pow(j);
        let k_plus_1_to_j = compose_affine(&mono_j, &Rational::from(1), &Rational::from(1));
        let term_a = a * &k_plus_1_to_j;
        let term_b = b_at_k_minus_1 * &mono_j;
        basis_polys.push((&term_a - &term_b).trim());
    }

    let mut max_deg = c.degree().max(0) as usize;
    for p in &basis_polys {
        max_deg = max_deg.max(p.degree().max(0) as usize);
    }

    let n_eq = max_deg + 1;
    let n_var = deg_x + 1;
    let mut mat = vec![vec![Rational::from(0); n_var]; n_eq];
    let mut rhs = vec![Rational::from(0); n_eq];

    for m in 0..n_eq {
        for j in 0..n_var {
            mat[m][j] = poly_nth_coeff(&basis_polys[j], m as i32);
        }
        rhs[m] = poly_nth_coeff(c, m as i32);
    }

    let sol = rational_gaussian_solve(mat, rhs)?;
    let coeffs: Vec<Rational> = sol[..n_var].to_vec();
    let x = RatUniPoly { coeffs }.trim();
    if x.is_zero() {
        return None;
    }

    let mut lhs = RatUniPoly::zero();
    for j in 0..=deg_x {
        let mono_j = x_pow(j);
        let k_plus_1_to_j = compose_affine(&mono_j, &Rational::from(1), &Rational::from(1));
        let term_a = a * &k_plus_1_to_j;
        let term_b = b_at_k_minus_1 * &mono_j;
        let p_j = (&term_a - &term_b).trim();
        let cj = poly_nth_coeff(&x, j as i32);
        lhs = (&lhs + &scale_poly(&p_j, &cj)).trim();
    }
    if (&lhs - c).trim().is_zero() {
        Some(x)
    } else {
        None
    }
}

/// Rational `R(k)` such that for hypergeometric `F(k)` with `F(k+1)/F(k)=r(k)`,
/// `G(k)=R(k)·F(k)` satisfies `G(k+1)-G(k)=F(k)` when such `R` exists.
pub fn gosper_certificate(r: &RatFunc) -> Option<RatFunc> {
    let r = r.clone().normalize();
    if r.num.is_zero() {
        return Some(RatFunc::zero());
    }
    let (a, b, c) = gosper_normal_form(r.num.clone(), r.den.clone());
    let b_eq = compose_affine(&b, &Rational::from(1), &Rational::from(-1));

    let mut candidates = gosper_degree_candidates(&a, &b_eq, &c);
    if candidates.is_empty() {
        candidates.push(15);
    }
    let max_d = *candidates.iter().max().unwrap() as usize;

    let max_scan = max_d.max(25);
    let x = solve_gosper_poly(&a, &b_eq, &c, max_scan)?;

    let num = &b_eq * &x;
    Some(
        RatFunc {
            num,
            den: c.clone(),
        }
        .normalize(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gosper_k_factorial_ratio() {
        let num = compose_affine(&RatUniPoly::x(), &Rational::from(1), &Rational::from(1));
        let num = &num * &num;
        let den = RatUniPoly::x();
        let r = RatFunc { num, den }.normalize();
        let cert = gosper_certificate(&r).expect("Gosper certificate exists");
        assert!(!cert.num.is_zero());
    }
}
