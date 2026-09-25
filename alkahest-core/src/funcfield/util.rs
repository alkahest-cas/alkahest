//! Exact ℚ[x] and ℚ-linear-algebra helpers used by the function-field layer.
//!
//! Everything here is exact: `rug::Rational` throughout, no floating point and
//! no truncation that is not explicit in a `k`-term series bound.  The
//! polynomial type is [`QPoly`] — the dense `Vec<Rational>`, lowest degree
//! first, that `integrate::risch` already uses — so that this module and the
//! Risch machinery it calls into share one representation.

use rug::{Integer, Rational};

use crate::integrate::risch::poly_rde::{degree, poly_deriv, poly_mul, trim, QPoly};
use crate::integrate::risch::rational_rde::{poly_div_exact, poly_gcd, poly_monic};

/// `p(x₀)` by Horner.
pub(crate) fn eval(p: &QPoly, x0: &Rational) -> Rational {
    let mut acc = Rational::from(0);
    for c in p.iter().rev() {
        acc = acc * x0 + c;
    }
    acc
}

/// `true` when `p` is the zero polynomial.
pub(crate) fn is_zero(p: &QPoly) -> bool {
    degree(p) < 0
}

/// The constant polynomial `c`.
pub(crate) fn constant(c: Rational) -> QPoly {
    trim(vec![c])
}

/// The linear polynomial `x − α`.
pub(crate) fn linear(alpha: &Rational) -> QPoly {
    trim(vec![-alpha.clone(), Rational::from(1)])
}

/// Synthetic division of `c` by `x − α`: returns `(quotient, remainder)`.
fn synth_div(c: &[Rational], alpha: &Rational) -> (QPoly, Rational) {
    let m = c.len();
    if m == 0 {
        return (Vec::new(), Rational::from(0));
    }
    if m == 1 {
        return (Vec::new(), c[0].clone());
    }
    let mut q = vec![Rational::from(0); m - 1];
    q[m - 2] = c[m - 1].clone();
    for i in (1..m - 1).rev() {
        q[i - 1] = c[i].clone() + Rational::from(&q[i] * alpha);
    }
    let r = c[0].clone() + Rational::from(&q[0] * alpha);
    (q, r)
}

/// Taylor shift: `q(t) = p(α + t)`, returned in `t`.
///
/// Computed by repeated synthetic division — exact, and `O(deg² )` rational
/// operations with no binomial coefficients to overflow.
pub(crate) fn taylor_shift(p: &QPoly, alpha: &Rational) -> QPoly {
    let p = trim(p.clone());
    if p.is_empty() {
        return Vec::new();
    }
    let n = p.len();
    let mut out = vec![Rational::from(0); n];
    let mut c: QPoly = p;
    for slot in out.iter_mut().take(n) {
        if c.is_empty() {
            break;
        }
        let (q, r) = synth_div(&c, alpha);
        *slot = r;
        c = q;
    }
    trim(out)
}

/// Multiply two truncated power series, keeping `k` coefficients (`t⁰..t^{k−1}`).
pub(crate) fn series_mul(a: &[Rational], b: &[Rational], k: usize) -> Vec<Rational> {
    let mut out = vec![Rational::from(0); k];
    for (i, ai) in a.iter().enumerate().take(k) {
        if *ai == 0 {
            continue;
        }
        for (j, bj) in b.iter().enumerate().take(k - i) {
            if *bj == 0 {
                continue;
            }
            out[i + j] += Rational::from(ai * bj);
        }
    }
    out
}

/// A `QPoly` viewed as a truncated series with `k` coefficients.
pub(crate) fn series_of_poly(p: &QPoly, k: usize) -> Vec<Rational> {
    let mut out = vec![Rational::from(0); k];
    for (i, slot) in out.iter_mut().enumerate() {
        if let Some(c) = p.get(i) {
            *slot = c.clone();
        }
    }
    out
}

/// `√r` when `r` is the square of a rational, else `None`.
///
/// The positive root is returned.  A negative `r` is never a square in ℚ.
pub(crate) fn rational_sqrt(r: &Rational) -> Option<Rational> {
    if *r < 0 {
        return None;
    }
    if *r == 0 {
        return Some(Rational::from(0));
    }
    let num = r.numer().clone();
    let den = r.denom().clone();
    if !num.is_perfect_square() || !den.is_perfect_square() {
        return None;
    }
    let sn: Integer = num.sqrt();
    let sd: Integer = den.sqrt();
    Some(Rational::from((sn, sd)))
}

/// Yun's squarefree decomposition of a non-zero `a ∈ ℚ[x]`.
///
/// Returns `(sqfree, sq)` with `a = sqfree · sq²`, `sqfree` **squarefree** and
/// `sq` monic.  The leading coefficient of `a` is carried by `sqfree`.
///
/// Used to normalise `y² = a(x)` to `Y² = sqfree(x)` with `Y = y / sq(x)`, a
/// ℚ(x)-isomorphism of the function field.
pub(crate) fn squarefree_split(a: &QPoly) -> (QPoly, QPoly) {
    let a = trim(a.clone());
    debug_assert!(!is_zero(&a));
    let d = degree(&a);
    if d <= 0 {
        return (a, constant(Rational::from(1)));
    }
    let lc = a[d as usize].clone();
    let f = poly_monic(&a);
    let fp = poly_deriv(&f);
    let a0 = poly_gcd(&f, &fp);
    let mut b = poly_div_exact(&f, &a0);
    let mut c = poly_div_exact(&fp, &a0);
    let mut dd = crate::integrate::risch::rational_rde::poly_sub(&c, &poly_deriv(&b));

    let mut odd_part = constant(Rational::from(1));
    let mut square_root_part = constant(Rational::from(1));
    let mut i: usize = 1;
    // `deg b` strictly decreases each round once a factor is peeled, and the
    // loop is additionally capped by the degree to keep it total.
    while degree(&b) > 0 && i <= d as usize + 1 {
        let ai = poly_gcd(&b, &dd);
        if i % 2 == 1 {
            odd_part = poly_mul(&odd_part, &ai);
        }
        for _ in 0..(i / 2) {
            square_root_part = poly_mul(&square_root_part, &ai);
        }
        b = poly_div_exact(&b, &ai);
        c = poly_div_exact(&dd, &ai);
        dd = crate::integrate::risch::rational_rde::poly_sub(&c, &poly_deriv(&b));
        i += 1;
    }
    let sqfree = poly_mul(&odd_part, &constant(lc));
    (trim(sqfree), trim(poly_monic(&square_root_part)))
}

/// Write `k = s·m²` with `s` squarefree and `m > 0`, keeping the sign on `s`.
///
/// The trial division is capped: past the cap a residual square factor is left
/// on `s`, which costs canonicality (two presentations of the same field may
/// then compare unequal) but never correctness — `s·m²` always equals `k`.
fn integer_squarefree_part(k: &Integer) -> (Integer, Integer) {
    const CAP: u32 = 1_000_000;
    let neg = *k < 0;
    let mut n = k.clone().abs();
    let mut m = Integer::from(1);
    let mut d = Integer::from(2);
    while Integer::from(&d * &d) <= n && d <= CAP {
        let dd = Integer::from(&d * &d);
        while n.is_divisible(&dd) {
            n /= &dd;
            m *= &d;
        }
        d += 1;
    }
    (if neg { -n } else { n }, m)
}

/// Canonicalise `y² = a(x)` modulo the substitutions `y ↦ c·y` that keep it a
/// polynomial model.
///
/// Returns `(a_norm, c)` with `a_norm = c²·a`, so `Y = c·y` satisfies
/// `Y² = a_norm(x)`.  The normal form is `a_norm = s·A(x)` with `A ∈ ℤ[x]`
/// primitive and positive-leading, and `s` a **squarefree integer**.
///
/// The squarefree `s` is kept, not discarded: `y² = 2(x³ − x)` is a quadratic
/// **twist** of `y² = x³ − x`, a genuinely different function field over ℚ, and
/// scaling it away would identify two curves that are not isomorphic.  Only
/// *square* constants — which `y ↦ c·y` really does absorb — are removed.
pub(crate) fn content_normalise(a: &QPoly) -> (QPoly, Rational) {
    let a = trim(a.clone());
    if is_zero(&a) {
        return (a, Rational::from(1));
    }
    let mut q = Integer::from(1);
    for c in &a {
        q = q.lcm(c.denom());
    }
    let ints: Vec<Integer> = a
        .iter()
        .map(|c| Rational::from(c * &q).numer().clone())
        .collect();
    let mut g = Integer::from(0);
    for c in &ints {
        g = g.gcd(c);
    }
    if g == 0 {
        return (a, Rational::from(1));
    }
    let mut prim: Vec<Integer> = ints.iter().map(|c| Integer::from(c / &g)).collect();
    let mut p = g;
    if *prim
        .last()
        .expect("non-zero polynomial has a leading coefficient")
        < 0
    {
        for c in prim.iter_mut() {
            *c = -c.clone();
        }
        p = -p;
    }
    // Reduce p/q to lowest terms; then a = (p/q)·prim and (q·y)² = p·q·prim.
    let r = Rational::from((p, q));
    let (p, q) = (r.numer().clone(), r.denom().clone());
    let k = p * q.clone();
    let (s, m) = integer_squarefree_part(&k);
    let a_norm: QPoly = prim.iter().map(|c| Rational::from(c * &s)).collect();
    (trim(a_norm), Rational::from((q, m)))
}

/// Rational roots of `p` with multiplicity, or `None` if `p` does not split
/// completely over ℚ.
///
/// A `None` is a refusal ("could not establish"), never a proof of
/// irreducibility — the underlying divisor search is capped.  Callers turn it
/// into [`super::FunctionFieldError::NonRationalSupport`].
pub(crate) fn rational_roots(p: &QPoly) -> Option<Vec<(Rational, usize)>> {
    let (_lc, roots) = crate::calculus::asymptotic_common::split_rational_roots(p)?;
    let mut out: Vec<(Rational, usize)> = Vec::new();
    for r in roots {
        if let Some(slot) = out.iter_mut().find(|(v, _)| *v == r) {
            slot.1 += 1;
        } else {
            out.push((r, 1));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Some(out)
}

/// `v_α(p)` — the multiplicity of `x − α` in `p`, or `None` when `p = 0`
/// (where the valuation is `+∞`).
pub(crate) fn valuation_at(p: &QPoly, alpha: &Rational) -> Option<usize> {
    if is_zero(p) {
        return None;
    }
    let mut c = trim(p.clone());
    let mut k = 0usize;
    loop {
        let (q, r) = synth_div(&c, alpha);
        if r != 0 {
            return Some(k);
        }
        k += 1;
        c = trim(q);
        if is_zero(&c) {
            // Cannot happen for a non-zero `p`: the degree bounds the loop.
            return Some(k);
        }
    }
}

/// Divide `p` by `(x − α)^k`, which must divide it exactly.
pub(crate) fn deflate(p: &QPoly, alpha: &Rational, k: usize) -> QPoly {
    let mut c = trim(p.clone());
    for _ in 0..k {
        if is_zero(&c) {
            return Vec::new();
        }
        let (q, _r) = synth_div(&c, alpha);
        debug_assert!(_r == 0, "deflate: (x - alpha) did not divide exactly");
        c = trim(q);
    }
    c
}

/// Reduced row echelon form of `m` in place; returns the pivot column of each
/// non-zero row, in order.
fn rref(m: &mut [Vec<Rational>], ncols: usize) -> Vec<usize> {
    let mut pivots = Vec::new();
    let mut row = 0usize;
    for col in 0..ncols {
        let Some(sel) = (row..m.len()).find(|&r| m[r][col] != 0) else {
            continue;
        };
        m.swap(row, sel);
        let inv = Rational::from(1) / m[row][col].clone();
        for c in m[row].iter_mut().take(ncols).skip(col) {
            *c *= &inv;
        }
        for r in 0..m.len() {
            if r == row || m[r][col] == 0 {
                continue;
            }
            let factor = m[r][col].clone();
            let pivot_row: Vec<Rational> = m[row][col..ncols].to_vec();
            for (c, pv) in m[r][col..ncols].iter_mut().zip(pivot_row.iter()) {
                *c -= Rational::from(&factor * pv);
            }
        }
        pivots.push(col);
        row += 1;
        if row == m.len() {
            break;
        }
    }
    pivots
}

/// A basis of the kernel of the `rows × ncols` rational matrix `m`.
///
/// Exact Gaussian elimination over ℚ.  The basis is in the canonical
/// "one free variable set to 1" form, ordered by free column, which makes the
/// Riemann–Roch basis it produces deterministic.
pub(crate) fn nullspace(mut m: Vec<Vec<Rational>>, ncols: usize) -> Vec<Vec<Rational>> {
    if ncols == 0 {
        return Vec::new();
    }
    if m.is_empty() {
        // No constraints: the whole space.
        return (0..ncols)
            .map(|i| {
                let mut v = vec![Rational::from(0); ncols];
                v[i] = Rational::from(1);
                v
            })
            .collect();
    }
    let pivots = rref(&mut m, ncols);
    let free: Vec<usize> = (0..ncols).filter(|c| !pivots.contains(c)).collect();
    let mut basis = Vec::with_capacity(free.len());
    for &fc in &free {
        let mut v = vec![Rational::from(0); ncols];
        v[fc] = Rational::from(1);
        for (r, &pc) in pivots.iter().enumerate() {
            v[pc] = -m[r][fc].clone();
        }
        basis.push(v);
    }
    basis
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        trim(cs.iter().map(|&c| Rational::from(c)).collect())
    }

    #[test]
    fn taylor_shift_matches_direct_evaluation() {
        // p = x³ − 2x + 5, shifted to α = 3.
        let p = qp(&[5, -2, 0, 1]);
        let alpha = Rational::from(3);
        let q = taylor_shift(&p, &alpha);
        // q(t) = p(3 + t) — check at several t.
        for t in [-2i64, 0, 1, 7] {
            let tt = Rational::from(t);
            let lhs = eval(&q, &tt);
            let rhs = eval(&p, &(Rational::from(3) + &tt));
            assert_eq!(lhs, rhs, "taylor shift disagrees at t = {t}");
        }
    }

    #[test]
    fn squarefree_split_peels_the_square_part() {
        // a = x³(x − 1) = x⁴ − x³  ⇒  sqfree = x(x−1), sq = x.
        let a = qp(&[0, 0, 0, -1, 1]);
        let (s, b) = squarefree_split(&a);
        assert_eq!(s, qp(&[0, -1, 1]), "squarefree part");
        assert_eq!(b, qp(&[0, 1]), "square root part");
        // And the identity a = s·b² holds exactly.
        assert_eq!(trim(poly_mul(&s, &poly_mul(&b, &b))), trim(a));
    }

    #[test]
    fn squarefree_split_is_identity_on_a_squarefree_input() {
        let a = qp(&[-1, 0, 0, 1]); // x³ − 1, squarefree
        let (s, b) = squarefree_split(&a);
        assert_eq!(s, a);
        assert_eq!(b, qp(&[1]));
    }

    #[test]
    fn squarefree_split_keeps_the_leading_coefficient() {
        // 12·(x−1)²·(x+2) — lc must end up on the squarefree part.
        let base = poly_mul(&qp(&[1, -2, 1]), &qp(&[2, 1]));
        let a: QPoly = base.iter().map(|c| Rational::from(c * 12)).collect();
        let (s, b) = squarefree_split(&trim(a.clone()));
        assert_eq!(trim(poly_mul(&s, &poly_mul(&b, &b))), trim(a));
        assert_eq!(b, qp(&[-1, 1]));
    }

    #[test]
    fn content_normalise_removes_square_constants_only() {
        // 16·(x³ − x)  ⇒  x³ − x, with Y = y/4.
        let (n, c) = content_normalise(&qp(&[0, -16, 0, 16]));
        assert_eq!(n, qp(&[0, -1, 0, 1]));
        assert_eq!(c, Rational::from((1, 4)));
        // A squarefree constant is a genuine quadratic twist and is kept.
        let (n, c) = content_normalise(&qp(&[0, -2, 0, 2]));
        assert_eq!(n, qp(&[0, -2, 0, 2]));
        assert_eq!(c, Rational::from(1));
        // Denominators are cleared onto the curve, not the substitution alone.
        let a: QPoly = vec![
            Rational::from(0),
            Rational::from(0),
            Rational::from(0),
            Rational::from((1, 3)),
        ];
        let (n, c) = content_normalise(&a);
        assert_eq!(n, qp(&[0, 0, 0, 3]));
        assert_eq!(c, Rational::from(3));
        // And the defining identity a_norm = c²·a holds in every case.
        for a in [qp(&[0, -16, 0, 16]), qp(&[0, -2, 0, 2]), qp(&[5, 0, 45])] {
            let (n, c) = content_normalise(&a);
            let want: QPoly = a
                .iter()
                .map(|v| Rational::from(v * &Rational::from(&c * &c)))
                .collect();
            assert_eq!(n, trim(want));
        }
    }

    #[test]
    fn integer_squarefree_part_splits_correctly() {
        for k in [1i64, 4, 12, -18, 72, -1, 2, 2025] {
            let ki = Integer::from(k);
            let (s, m) = integer_squarefree_part(&ki);
            assert_eq!(Integer::from(&s * &Integer::from(&m * &m)), ki, "k = {k}");
            assert!(m > 0);
        }
        assert_eq!(
            integer_squarefree_part(&Integer::from(72)),
            (Integer::from(2), Integer::from(6))
        );
    }

    #[test]
    fn rational_sqrt_only_accepts_squares() {
        assert_eq!(
            rational_sqrt(&Rational::from((9, 4))),
            Some(Rational::from((3, 2)))
        );
        assert_eq!(rational_sqrt(&Rational::from(2)), None);
        assert_eq!(rational_sqrt(&Rational::from(-1)), None);
        assert_eq!(rational_sqrt(&Rational::from(0)), Some(Rational::from(0)));
    }

    #[test]
    fn rational_roots_refuses_an_irrational_factor() {
        // x² − 2 has no rational root.
        assert!(rational_roots(&qp(&[-2, 0, 1])).is_none());
        // (x−1)²(x+3) splits.
        let p = poly_mul(&qp(&[1, -2, 1]), &qp(&[3, 1]));
        let rs = rational_roots(&trim(p)).unwrap();
        assert_eq!(rs, vec![(Rational::from(-3), 1), (Rational::from(1), 2)]);
    }

    #[test]
    fn nullspace_of_a_known_matrix() {
        // [[1, 1, 1]] ⇒ kernel is 2-dimensional.
        let m = vec![vec![
            Rational::from(1),
            Rational::from(1),
            Rational::from(1),
        ]];
        let ker = nullspace(m, 3);
        assert_eq!(ker.len(), 2);
        for v in &ker {
            let s: Rational = v.iter().fold(Rational::from(0), |acc, c| acc + c);
            assert_eq!(s, 0);
        }
    }

    #[test]
    fn nullspace_of_a_full_rank_square_matrix_is_trivial() {
        let m = vec![
            vec![Rational::from(1), Rational::from(0)],
            vec![Rational::from(0), Rational::from(1)],
        ];
        assert!(nullspace(m, 2).is_empty());
    }

    #[test]
    fn nullspace_with_no_rows_is_everything() {
        assert_eq!(nullspace(Vec::new(), 3).len(), 3);
    }

    #[test]
    fn valuation_and_deflate_are_inverse() {
        // (x − 2)³·(x + 1)
        let base = poly_mul(&qp(&[-2, 1]), &poly_mul(&qp(&[-2, 1]), &qp(&[-2, 1])));
        let p = trim(poly_mul(&base, &qp(&[1, 1])));
        let a = Rational::from(2);
        assert_eq!(valuation_at(&p, &a), Some(3));
        assert_eq!(deflate(&p, &a, 3), qp(&[1, 1]));
        assert_eq!(valuation_at(&p, &Rational::from(0)), Some(0));
        assert_eq!(valuation_at(&Vec::new(), &a), None);
    }

    #[test]
    fn series_mul_truncates() {
        // (1 + t)·(1 + t) = 1 + 2t + t², keep 2 terms.
        let a = vec![Rational::from(1), Rational::from(1)];
        let out = series_mul(&a, &a, 2);
        assert_eq!(out, vec![Rational::from(1), Rational::from(2)]);
    }
}
