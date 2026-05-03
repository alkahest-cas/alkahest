//! Smith normal decomposition over ℤ — port of SymPy `_smith_normal_decomp` (`smith_normal_decomp`).
//!
//! Returns `(S, U, V)` with `S == U * M * V` and `S` rectangular-diagonal in Smith form.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::cmp_owned
)]

use rug::Complete;
use rug::Integer;
use std::cmp::Ordering;

fn zz_div_rem(a: &Integer, b: &Integer) -> (Integer, Integer) {
    if b.cmp0() == Ordering::Equal {
        panic!("division by zero in Smith step");
    }
    a.div_rem_floor_ref(b).complete()
}

fn zz_gcdex(a: &Integer, b: &Integer) -> (Integer, Integer, Integer) {
    let (g, s, t) = a.clone().extended_gcd(b.clone(), Integer::new());
    (s, t, g)
}

fn zz_exquo(a: &Integer, g: &Integer) -> Integer {
    let (q, r) = zz_div_rem(a, g);
    if r.cmp0() != Ordering::Equal {
        panic!("exact quotient required in Smith step");
    }
    q
}

fn zz_canonical_unit(a: &Integer) -> Integer {
    match a.cmp0() {
        Ordering::Less => Integer::from(-1),
        _ => Integer::from(1),
    }
}

fn add_columns(
    m: &mut [Vec<Integer>],
    i: usize,
    j: usize,
    a: &Integer,
    b: &Integer,
    c: &Integer,
    d: &Integer,
) {
    for row in m.iter_mut() {
        let e = row[i].clone();
        row[i] = a.clone() * &e + b.clone() * &row[j];
        row[j] = c.clone() * e + d.clone() * &row[j];
    }
}

fn add_rows(
    m: &mut [Vec<Integer>],
    i: usize,
    j: usize,
    a: &Integer,
    b: &Integer,
    c: &Integer,
    d: &Integer,
) {
    let n = m[0].len();
    for k in 0..n {
        let e = m[i][k].clone();
        m[i][k] = a.clone() * &e + b.clone() * &m[j][k];
        m[j][k] = c.clone() * e + d.clone() * &m[j][k];
    }
}

fn mat_mul_square(lhs: &[Vec<Integer>], rhs: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
    let n = lhs.len();
    let mut out = vec![vec![Integer::from(0); n]; n];
    for i in 0..n {
        for k in 0..n {
            let aik = lhs[i][k].clone();
            if aik.cmp0() == Ordering::Equal {
                continue;
            }
            for j in 0..n {
                out[i][j] += &aik * &rhs[k][j];
            }
        }
    }
    out
}

fn add_rows_s(
    m: &mut [Vec<Integer>],
    s: &mut Option<Vec<Vec<Integer>>>,
    full: bool,
    i: usize,
    j: usize,
    a: &Integer,
    b: &Integer,
    c: &Integer,
    d: &Integer,
) {
    add_rows(m, i, j, a, b, c, d);
    if full {
        if let Some(ss) = s.as_mut() {
            add_rows(ss, i, j, a, b, c, d);
        }
    }
}

fn add_columns_t(
    m: &mut [Vec<Integer>],
    t: &mut Option<Vec<Vec<Integer>>>,
    full: bool,
    i: usize,
    j: usize,
    a: &Integer,
    b: &Integer,
    c: &Integer,
    d: &Integer,
) {
    add_columns(m, i, j, a, b, c, d);
    if full {
        if let Some(tt) = t.as_mut() {
            add_columns(tt, i, j, a, b, c, d);
        }
    }
}

fn smith_decomp_rec(
    mut m: Vec<Vec<Integer>>,
    rows: usize,
    cols: usize,
    full: bool,
) -> (
    Vec<Integer>,
    Option<Vec<Vec<Integer>>>,
    Option<Vec<Vec<Integer>>>,
) {
    let zero = Integer::from(0);
    let one = Integer::from(1);

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

    let mut s_opt: Option<Vec<Vec<Integer>>> = if full { Some(eye(rows)) } else { None };
    let mut t_opt: Option<Vec<Vec<Integer>>> = if full { Some(eye(cols)) } else { None };

    let ind_row: Vec<usize> = (0..rows).filter(|&i| m[i][0] != zero).collect();
    if !ind_row.is_empty() && ind_row[0] != 0 {
        let ix = ind_row[0];
        m.swap(0, ix);
        if let Some(ss) = s_opt.as_mut() {
            ss.swap(0, ix);
        }
    } else {
        let ind_col: Vec<usize> = (0..cols).filter(|&j| m[0][j] != zero).collect();
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
        let row_nz = (1..cols).any(|j| m[0][j] != zero);
        let col_nz = (1..rows).any(|i| m[i][0] != zero);
        if !row_nz && !col_nz {
            break;
        }
        if m[0][0] == zero {
            break;
        }

        let mut pivot = m[0][0].clone();
        for j in 1..rows {
            if m[j][0] == zero {
                continue;
            }
            let (d, r) = zz_div_rem(&m[j][0], &pivot);
            if r == zero {
                let neg_d = -d.clone();
                add_rows_s(&mut m, &mut s_opt, full, 0, j, &one, &zero, &neg_d, &one);
            } else {
                let (a, b, g) = zz_gcdex(&pivot, &m[j][0]);
                let d0 = zz_exquo(&m[j][0], &g);
                let dj = zz_exquo(&pivot, &g);
                let neg_dj = -dj.clone();
                add_rows_s(&mut m, &mut s_opt, full, 0, j, &a, &b, &d0, &neg_dj);
                pivot = g;
            }
        }

        if m[0][0] == zero {
            break;
        }

        pivot = m[0][0].clone();
        for j in 1..cols {
            if m[0][j] == zero {
                continue;
            }
            let (d, r) = zz_div_rem(&m[0][j], &pivot);
            if r == zero {
                let neg_d = -d.clone();
                add_columns_t(&mut m, &mut t_opt, full, 0, j, &one, &zero, &neg_d, &one);
            } else {
                let (a, b, g) = zz_gcdex(&pivot, &m[0][j]);
                let d0 = zz_exquo(&m[0][j], &g);
                let dj = zz_exquo(&pivot, &g);
                let neg_dj = -dj.clone();
                add_columns_t(&mut m, &mut t_opt, full, 0, j, &a, &b, &d0, &neg_dj);
                pivot = g;
            }
        }
    }

    if m[0][0] != zero {
        let c = zz_canonical_unit(&m[0][0]);
        if c != one {
            m[0][0] *= &c;
            if let Some(ss) = s_opt.as_mut() {
                for e in ss[0].iter_mut() {
                    *e *= &c;
                }
            }
        }
    }

    let invs: Vec<Integer>;
    if rows == 1 || cols == 1 {
        invs = vec![];
    } else {
        let lower_right: Vec<Vec<Integer>> = m[1..].iter().map(|r| r[1..].to_vec()).collect();
        let (inner_invs, s2_small, t2_small) =
            smith_decomp_rec(lower_right, rows - 1, cols - 1, full);

        invs = inner_invs;

        if full {
            let s_small = s2_small.unwrap();
            let t_small = t2_small.unwrap();

            let mut s2: Vec<Vec<Integer>> = Vec::with_capacity(rows);
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

    let result: Vec<Integer>;
    if m[0][0] != zero {
        let mut res = vec![m[0][0].clone()];
        res.extend(invs);

        let mut i = 0;
        while i + 1 < res.len() {
            let a = res[i].clone();
            let b = res[i + 1].clone();
            let (_, rem) = zz_div_rem(&b, &a);
            if b.cmp0() != Ordering::Equal && rem != zero {
                let (x, y, d) = zz_gcdex(&a, &b);
                let alpha = zz_exquo(&a, &d);
                let beta = zz_exquo(&b, &d);
                if full {
                    let neg_alpha = -alpha.clone();
                    let neg_one = -one.clone();
                    let neg_beta = -beta.clone();
                    if let Some(ss) = s_opt.as_mut() {
                        add_rows(ss, i, i + 1, &one, &zero, &x, &one);
                        add_rows(ss, i, i + 1, &one, &neg_alpha, &zero, &one);
                        add_rows(ss, i, i + 1, &zero, &one, &neg_one, &zero);
                    }
                    if let Some(tt) = t_opt.as_mut() {
                        add_columns(tt, i, i + 1, &one, &y, &zero, &one);
                        // Matches SymPy: col j ← (−β)·col_i + col_j (not β·col_j).
                        add_columns(tt, i, i + 1, &one, &zero, &neg_beta, &one);
                    }
                }
                res[i] = d;
                res[i + 1] = b * &alpha;
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

/// Returns `(S, U, V)` with `S == U * M * V`.
pub(super) fn smith_normal_decomp(
    m: Vec<Vec<Integer>>,
) -> (Vec<Vec<Integer>>, Vec<Vec<Integer>>, Vec<Vec<Integer>>) {
    let rows = m.len();
    let cols = if rows == 0 { 0 } else { m[0].len() };
    let (invs, s, t) = smith_decomp_rec(m, rows, cols, true);
    let u = s.unwrap();
    let v = t.unwrap();
    let mut diag = vec![vec![Integer::from(0); cols]; rows];
    let lim = invs.len().min(rows.min(cols));
    for i in 0..lim {
        diag[i][i] = invs[i].clone();
    }
    debug_assert!(
        diag.iter().enumerate().all(|(i, row)| row
            .iter()
            .enumerate()
            .all(|(j, e)| i == j || *e == Integer::from(0))),
        "internal: Smith form matrix has off-diagonal nonzero"
    );
    (diag, u, v)
}
