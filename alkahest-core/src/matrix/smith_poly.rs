//! Smith normal form over `ℚ[x]` and Hermite column form (for `PolyMatrixQ`).

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::cmp_owned
)]

use super::{PolyMatrixQ, RatUniPoly};
use rug::Rational;

fn poly_canonical(p: &RatUniPoly) -> RatUniPoly {
    if p.is_zero() {
        RatUniPoly::one()
    } else {
        RatUniPoly::constant(Rational::from(1) / p.leading_coeff())
    }
}

fn add_columns(
    m: &mut [Vec<RatUniPoly>],
    i: usize,
    j: usize,
    a: &RatUniPoly,
    b: &RatUniPoly,
    c: &RatUniPoly,
    d: &RatUniPoly,
) {
    for row in m.iter_mut() {
        let e = row[i].clone();
        row[i] = (&(a * &e) + &(b * &row[j])).trim();
        row[j] = (&(c * &e) + &(d * &row[j])).trim();
    }
}

fn add_rows(
    m: &mut [Vec<RatUniPoly>],
    i: usize,
    j: usize,
    a: &RatUniPoly,
    b: &RatUniPoly,
    c: &RatUniPoly,
    d: &RatUniPoly,
) {
    let n = m[0].len();
    for k in 0..n {
        let e = m[i][k].clone();
        m[i][k] = (&(a * &e) + &(b * &m[j][k])).trim();
        m[j][k] = (&(c * &e) + &(d * &m[j][k])).trim();
    }
}

fn mat_mul_square(lhs: &[Vec<RatUniPoly>], rhs: &[Vec<RatUniPoly>]) -> Vec<Vec<RatUniPoly>> {
    let n = lhs.len();
    let mut out = vec![vec![RatUniPoly::zero(); n]; n];
    for i in 0..n {
        for k in 0..n {
            if lhs[i][k].is_zero() {
                continue;
            }
            for j in 0..n {
                let p = &lhs[i][k] * &rhs[k][j];
                out[i][j] = (&out[i][j] + &p).trim();
            }
        }
    }
    out
}

fn add_rows_s(
    m: &mut [Vec<RatUniPoly>],
    s: &mut Option<Vec<Vec<RatUniPoly>>>,
    full: bool,
    i: usize,
    j: usize,
    a: &RatUniPoly,
    b: &RatUniPoly,
    c: &RatUniPoly,
    d: &RatUniPoly,
) {
    add_rows(m, i, j, a, b, c, d);
    if full {
        if let Some(ss) = s.as_mut() {
            add_rows(ss, i, j, a, b, c, d);
        }
    }
}

fn add_columns_t(
    m: &mut [Vec<RatUniPoly>],
    t: &mut Option<Vec<Vec<RatUniPoly>>>,
    full: bool,
    i: usize,
    j: usize,
    a: &RatUniPoly,
    b: &RatUniPoly,
    c: &RatUniPoly,
    d: &RatUniPoly,
) {
    add_columns(m, i, j, a, b, c, d);
    if full {
        if let Some(tt) = t.as_mut() {
            add_columns(tt, i, j, a, b, c, d);
        }
    }
}

fn smith_decomp_rec(
    mut m: Vec<Vec<RatUniPoly>>,
    rows: usize,
    cols: usize,
    full: bool,
) -> (
    Vec<RatUniPoly>,
    Option<Vec<Vec<RatUniPoly>>>,
    Option<Vec<Vec<RatUniPoly>>>,
) {
    let zero = RatUniPoly::zero();
    let one = RatUniPoly::one();

    let eye = |n: usize| {
        let mut id = vec![vec![zero.clone(); n]; n];
        for i in 0..n {
            id[i][i] = one.clone();
        }
        id
    };

    if rows == 0 || cols == 0 {
        if full {
            return (vec![], Some(eye(rows)), Some(eye(cols)));
        }
        return (vec![], None, None);
    }

    let mut s_opt: Option<Vec<Vec<RatUniPoly>>> = if full { Some(eye(rows)) } else { None };
    let mut t_opt: Option<Vec<Vec<RatUniPoly>>> = if full { Some(eye(cols)) } else { None };

    let ind_row: Vec<usize> = (0..rows).filter(|&i| !m[i][0].is_zero()).collect();
    if !ind_row.is_empty() && ind_row[0] != 0 {
        let ix = ind_row[0];
        m.swap(0, ix);
        if let Some(ss) = s_opt.as_mut() {
            ss.swap(0, ix);
        }
    } else {
        let ind_col: Vec<usize> = (0..cols).filter(|&j| !m[0][j].is_zero()).collect();
        if !ind_col.is_empty() && ind_col[0] != 0 {
            let jc = ind_col[0];
            for row in m.iter_mut() {
                row.swap(0, jc);
            }
            if let Some(tt) = t_opt.as_mut() {
                for row in tt.iter_mut() {
                    row.swap(0, jc);
                }
            }
        }
    }

    loop {
        let row_nz = (1..cols).any(|j| !m[0][j].is_zero());
        let col_nz = (1..rows).any(|i| !m[i][0].is_zero());
        if !row_nz && !col_nz {
            break;
        }
        if m[0][0].is_zero() {
            break;
        }

        let mut pivot = m[0][0].clone();
        for j in 1..rows {
            if m[j][0].is_zero() {
                continue;
            }
            let (d, r) = RatUniPoly::div_rem(&m[j][0], &pivot);
            if r.is_zero() {
                add_rows_s(&mut m, &mut s_opt, full, 0, j, &one, &zero, &-&d, &one);
            } else {
                let (a, b, g) = RatUniPoly::gcdex(&pivot, &m[j][0]);
                let d0 = m[j][0].exquo(&g);
                let dj = pivot.exquo(&g);
                add_rows_s(&mut m, &mut s_opt, full, 0, j, &a, &b, &d0, &-&dj);
                pivot = g;
            }
        }

        if m[0][0].is_zero() {
            break;
        }

        pivot = m[0][0].clone();
        for j in 1..cols {
            if m[0][j].is_zero() {
                continue;
            }
            let (d, r) = RatUniPoly::div_rem(&m[0][j], &pivot);
            if r.is_zero() {
                add_columns_t(&mut m, &mut t_opt, full, 0, j, &one, &zero, &-&d, &one);
            } else {
                let (a, b, g) = RatUniPoly::gcdex(&pivot, &m[0][j]);
                let d0 = m[0][j].exquo(&g);
                let dj = pivot.exquo(&g);
                add_columns_t(&mut m, &mut t_opt, full, 0, j, &a, &b, &d0, &-&dj);
                pivot = g;
            }
        }
    }

    if !m[0][0].is_zero() {
        let c = poly_canonical(&m[0][0]);
        if c != one {
            m[0][0] = (&m[0][0] * &c).trim();
            if let Some(ss) = s_opt.as_mut() {
                for e in ss[0].iter_mut() {
                    *e = (&*e * &c).trim();
                }
            }
        }
    }

    let invs: Vec<RatUniPoly>;
    if rows == 1 || cols == 1 {
        invs = vec![];
    } else {
        let lower_right: Vec<Vec<RatUniPoly>> = m[1..].iter().map(|r| r[1..].to_vec()).collect();
        let (inner_invs, s2_small, t2_small) =
            smith_decomp_rec(lower_right, rows - 1, cols - 1, full);

        invs = inner_invs;

        if full {
            let s_small = s2_small.unwrap();
            let t_small = t2_small.unwrap();

            let mut s2: Vec<Vec<RatUniPoly>> = Vec::with_capacity(rows);
            let mut row0 = vec![zero.clone(); rows];
            row0[0] = one.clone();
            s2.push(row0);
            for r in s_small {
                let mut nr = vec![zero.clone(); rows];
                for (k, val) in r.into_iter().enumerate() {
                    if k + 1 < rows {
                        nr[k + 1] = val;
                    }
                }
                s2.push(nr);
            }

            let mut t2 = vec![vec![zero.clone(); cols]; cols];
            t2[0][0] = one.clone();
            for i in 1..cols {
                for j in 1..cols {
                    t2[i][j] = t_small[i - 1][j - 1].clone();
                }
            }

            let s_old = s_opt.take().unwrap();
            let t_old = t_opt.take().unwrap();
            s_opt = Some(mat_mul_square(&s2, &s_old));
            t_opt = Some(mat_mul_square(&t_old, &t2));
        }
    }

    let result: Vec<RatUniPoly>;
    if !m[0][0].is_zero() {
        let mut res = vec![m[0][0].clone()];
        res.extend(invs);

        let mut i = 0;
        while i + 1 < res.len() {
            let a = res[i].clone();
            let b = res[i + 1].clone();
            let (_, rem) = RatUniPoly::div_rem(&b, &a);
            if !b.is_zero() && !rem.is_zero() {
                let (x, y, d) = RatUniPoly::gcdex(&a, &b);
                let alpha = a.exquo(&d);
                let beta = b.exquo(&d);
                if full {
                    if let Some(ss) = s_opt.as_mut() {
                        add_rows(ss, i, i + 1, &one, &zero, &x, &one);
                        add_rows(ss, i, i + 1, &one, &-&alpha, &zero, &one);
                        add_rows(ss, i, i + 1, &zero, &one, &-&one, &zero);
                    }
                    if let Some(tt) = t_opt.as_mut() {
                        add_columns(tt, i, i + 1, &one, &y, &zero, &one);
                        add_columns(tt, i, i + 1, &one, &zero, &zero, &-&beta);
                    }
                }
                res[i] = d;
                res[i + 1] = (&b * &alpha).trim();
            } else {
                break;
            }
            i += 1;
        }
        result = res;
    } else {
        if full {
            if let Some(ss) = s_opt.as_mut() {
                if rows > 1 {
                    let r0 = ss.remove(0);
                    ss.push(r0);
                }
            }
            if let Some(tt) = t_opt.as_mut() {
                if cols > 1 {
                    for row in tt.iter_mut() {
                        let c0 = row.remove(0);
                        row.push(c0);
                    }
                }
            }
        }
        let mut res = invs;
        res.push(m[0][0].clone());
        result = res;
    }

    (result, s_opt, t_opt)
}

pub(super) fn smith_normal_poly(m: &PolyMatrixQ) -> (PolyMatrixQ, PolyMatrixQ, PolyMatrixQ) {
    if m.rows == 0 || m.cols == 0 {
        let empty_s = PolyMatrixQ::shell(m.rows, m.cols);
        let ur = if m.rows == 0 {
            vec![]
        } else {
            identity_coeffs(m.rows)
        };
        let uc = if m.cols == 0 {
            vec![]
        } else {
            identity_coeffs(m.cols)
        };
        return (
            empty_s,
            PolyMatrixQ::from_nested(ur).unwrap(),
            PolyMatrixQ::from_nested(uc).unwrap(),
        );
    }
    let mat: Vec<Vec<RatUniPoly>> = (0..m.rows)
        .map(|i| (0..m.cols).map(|j| m.get(i, j).clone()).collect())
        .collect();
    let rows = m.rows;
    let cols = m.cols;
    let (invs, s, t) = smith_decomp_rec(mat, rows, cols, true);
    let u = s.unwrap();
    let v = t.unwrap();
    let mut diag = vec![vec![RatUniPoly::zero(); cols]; rows];
    let lim = invs.len().min(rows.min(cols));
    for i in 0..lim {
        diag[i][i] = invs[i].clone();
    }
    (
        PolyMatrixQ::from_nested(diag).unwrap(),
        PolyMatrixQ::from_nested(u).unwrap(),
        PolyMatrixQ::from_nested(v).unwrap(),
    )
}

fn poly_gcdex_hnf(a: &RatUniPoly, b: &RatUniPoly) -> (RatUniPoly, RatUniPoly, RatUniPoly) {
    let (mut x, mut y, g) = RatUniPoly::gcdex(a, b);
    if !a.is_zero() {
        let (_, r) = RatUniPoly::div_rem(b, a);
        if r.is_zero() {
            y = RatUniPoly::zero();
            let lc = a.leading_coeff();
            x = RatUniPoly::constant(Rational::from(1) / lc);
        }
    }
    (x, y, g)
}

fn flip_col(a: &mut [Vec<RatUniPoly>], v: &mut [Vec<RatUniPoly>], col: usize) {
    let m1 = RatUniPoly::constant(Rational::from(-1));
    for r in a.iter_mut() {
        r[col] = (&r[col] * &m1).trim();
    }
    for r in v.iter_mut() {
        r[col] = (&r[col] * &m1).trim();
    }
}

fn identity_coeffs(n: usize) -> Vec<Vec<RatUniPoly>> {
    let z = RatUniPoly::zero();
    let o = RatUniPoly::one();
    let mut m = vec![vec![z.clone(); n]; n];
    for i in 0..n {
        m[i][i] = o.clone();
    }
    m
}

/// Column-style Hermite normal form `A * V = H` (SymPy `_hermite_normal_form` over `ℚ[x]`).
pub(super) fn hermite_column_poly(m: &PolyMatrixQ) -> (PolyMatrixQ, PolyMatrixQ) {
    let rows = m.rows;
    let cols = m.cols;
    if rows == 0 || cols == 0 {
        let h = PolyMatrixQ::shell(rows, cols);
        let vid = if cols == 0 {
            vec![]
        } else {
            identity_coeffs(cols)
        };
        return (h, PolyMatrixQ::from_nested(vid).unwrap());
    }
    let mut a: Vec<Vec<RatUniPoly>> = (0..rows)
        .map(|i| (0..cols).map(|j| m.get(i, j).clone()).collect())
        .collect();
    let mut v = identity_coeffs(cols);
    let zero = RatUniPoly::zero();
    let one = RatUniPoly::one();
    let mut k = cols;
    for i in (0..rows).rev() {
        if k == 0 {
            break;
        }
        k -= 1;
        for j in (0..k).rev() {
            if a[i][j].is_zero() {
                continue;
            }
            let (u, vco, d) = poly_gcdex_hnf(&a[i][k], &a[i][j]);
            let rk = a[i][k].exquo(&d);
            let rj = a[i][j].exquo(&d);
            add_columns(&mut a, k, j, &u, &vco, &-&rj, &rk);
            add_columns(&mut v, k, j, &u, &vco, &-&rj, &rk);
        }
        let b = a[i][k].clone();
        if !b.is_zero() && b.leading_coeff() < Rational::from(0) {
            flip_col(&mut a, &mut v, k);
        }
        let b = a[i][k].clone();
        if b.is_zero() {
            k += 1;
        } else {
            for j in (k + 1)..cols {
                let (q, _) = RatUniPoly::div_rem(&a[i][j], &b);
                add_columns(&mut a, j, k, &one, &-&q, &zero, &one);
                add_columns(&mut v, j, k, &one, &-&q, &zero, &one);
            }
        }
    }
    (
        PolyMatrixQ::from_nested(a).unwrap(),
        PolyMatrixQ::from_nested(v).unwrap(),
    )
}
