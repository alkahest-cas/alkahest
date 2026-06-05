//! van Hoeij integral-basis **enlargement loop** (MB) — builds a global integral
//! basis of `ℚ(x)(y)`, `F(x,y)=0`, from the power basis by repeatedly dividing
//! out singular places.
//!
//! Following van Hoeij (1994 §2): start `b₀ = 1`; for `d = 1 … n−1`, set the
//! initial `b_d = y·b_{d−1}` (degree `d`) and enlarge it as long as some
//! `a = (a₀b₀ + ⋯ + a_{d−1}b_{d−1} + b_d)/(x−α)` is integral, where `α` is a
//! singular place (`(x−α)² | disc`).  The coefficients `aᵢ ∈ ℚ` are found by
//! substituting the Puiseux expansions at `α` into `a` and requiring every
//! **negative-power** coefficient (in the place's uniformizer) to vanish — a
//! `ℚ`-linear system.  Each proposed enlargement is then **gated by the exact
//! [`is_integral`] test**, so the result is sound regardless of Puiseux
//! truncation: a wrong proposal is rejected, never accepted.
//!
//! Scope: enlargements are proposed only at **rational** singular places, using
//! their rational Puiseux sheets.  Sheets with algebraic coefficients are simply
//! absent from the linear system, so a proposal there is under-constrained and
//! `is_integral` rejects it — the basis may then be non-maximal at such a place,
//! but is **never incorrect**.  Produces correct integral bases for radical
//! curves and rational-branch curves such as the nodal cubic `y² = x³ + x²`
//! (`{1, y/x}`).  Algebraic singular places (and algebraic sheets) are the
//! follow-up, building on `puiseux_at_zero_algebraic`.

use rug::{Integer, Rational};
use std::collections::BTreeMap;

use super::super::risch::alg_field::{AlgElem, AlgExtension, RatFn, RationalFunctionField};
use super::super::risch::number_field::CoeffField;
use super::super::risch::poly_rde::{degree, QPoly};
use super::integral_basis::{discriminant, is_integral, rational_singularities};
use crate::poly::puiseux::{puiseux_at, PuiseuxSeries};

/// A Laurent series in the place uniformizer `t = (x−α)^{1/e}` (integer
/// `t`-exponents), truncated.  Shared with [`super::residues`].
pub(super) type TS = BTreeMap<i64, Rational>;

/// Compute an integral basis `b₀, …, b_{n−1}` of `ℚ(x)(y)` over `ℚ[x]` for the
/// monic minimal polynomial `F = Σ f_coeffs[j] yʲ`.  Returns the basis as
/// `AlgElem`s (each verified integral), or `None` on failure.
///
/// Maximal when every singular place is rational with rational branches;
/// otherwise an integral (possibly non-maximal) basis — see the module docs.
pub fn integral_basis(f_coeffs: &[QPoly]) -> Option<Vec<AlgElem>> {
    let ext = AlgExtension::new(f_coeffs);
    let n = ext.degree() as usize;
    if n < 1 {
        return None;
    }
    let monos = to_monomials(f_coeffs);
    let disc = discriminant(f_coeffs);
    let sing = rational_singularities(&disc);

    let mut b: Vec<AlgElem> = vec![ext.from_int(1)]; // b₀ = 1
    for d in 1..n {
        let mut bd = ext.mul(&b[d - 1], &ext.generator()); // y·b_{d−1}
                                                           // Each real enlargement drops the discriminant by (x−α)²; bound the work.
        let cap = (degree(&disc).max(0) as usize + 2) * sing.len().max(1) + 4;
        let mut iters = 0;
        'enlarge: loop {
            if iters >= cap {
                break;
            }
            for alpha in &sing {
                if let Some(cand) = try_enlarge(&ext, &b, &bd, d, alpha, &monos, n) {
                    if !ext.elem_eq(&cand, &bd) && is_integral(f_coeffs, &cand) {
                        bd = cand;
                        iters += 1;
                        continue 'enlarge;
                    }
                }
            }
            break;
        }
        b.push(bd);
    }

    for bi in &b {
        if !is_integral(f_coeffs, bi) {
            return None;
        }
    }
    Some(b)
}

/// Try to enlarge `bd` at the place `x = α`: find `a₀…a_{d−1} ∈ ℚ` so that
/// `(Σ aᵢ bᵢ + bd)/(x−α)` is integral, returning that element (unverified —
/// caller gates with [`is_integral`]).  `None` if the place has algebraic
/// branches or no enlargement exists.
fn try_enlarge(
    ext: &AlgExtension,
    b: &[AlgElem],
    bd: &AlgElem,
    d: usize,
    alpha: &Rational,
    monos: &[(u32, u32, Rational)],
    n: usize,
) -> Option<AlgElem> {
    // Precision: enough Puiseux terms to resolve the coordinate denominators.
    let dmax = max_denom_degree(b, bd);
    let prec = (dmax + n + 3) as u32;
    // Substitute every (rational) sheet at α.  Missing algebraic sheets can only
    // *under*-constrain the system → an unverified proposal that `is_integral`
    // (the caller's gate) then rejects; soundness does not depend on completeness.
    let branches = puiseux_at(monos, alpha, prec);
    if branches.is_empty() {
        return None;
    }
    let emax = branches.iter().map(|s| s.ramification).max().unwrap_or(1) as i64;
    let u_bound = (prec as i64 + 3) * emax + 4;

    // Build the ℚ-linear system: for each branch and each negative-power
    // coefficient of (Σ aᵢ bᵢ + bd)/(x−α), the linear form in (a₀…a_{d−1}) is 0.
    let mut matrix: Vec<Vec<Rational>> = Vec::new();
    let mut rhs: Vec<Rational> = Vec::new();
    for s in &branches {
        let e = s.ramification as i64;
        let bts = branch_ts(s);
        // Series of each bᵢ (i<d) and of bd along this branch.
        let bi_ts: Vec<TS> = (0..d)
            .map(|i| elem_ts(&b[i], alpha, e, u_bound, &bts))
            .collect();
        let bd_ts = elem_ts(bd, alpha, e, u_bound, &bts);
        // Negative-power range: keys k < e (after dividing by (x−α)=t^e).
        let low = bi_ts
            .iter()
            .chain(std::iter::once(&bd_ts))
            .filter_map(|s| s.keys().next().copied())
            .min()
            .unwrap_or(0);
        for k in low..e {
            let row: Vec<Rational> = bi_ts
                .iter()
                .map(|s| s.get(&k).cloned().unwrap_or_else(|| Rational::from(0)))
                .collect();
            let r = -bd_ts.get(&k).cloned().unwrap_or_else(|| Rational::from(0));
            matrix.push(row);
            rhs.push(r);
        }
    }
    let a = gauss_solve(matrix, rhs, d)?;

    // candidate = (Σ aᵢ bᵢ + bd) · 1/(x−α).
    let mut num = bd.clone();
    for (i, ai) in a.iter().enumerate() {
        if *ai != 0 {
            num = ext.add(&num, &scale(&b[i], &RatFn::from_poly(&vec![ai.clone()])));
        }
    }
    let inv_lin = RatFn::new(
        vec![Rational::from(1)],
        vec![-alpha.clone(), Rational::from(1)],
    );
    Some(scale(&num, &inv_lin))
}

// ---------------------------------------------------------------------------
// Laurent-series-in-t arithmetic (over ℚ)
// ---------------------------------------------------------------------------

pub(super) fn branch_ts(s: &PuiseuxSeries) -> TS {
    let e = s.ramification as i64;
    let mut ts = TS::new();
    for (exp, c) in &s.terms {
        // exp has denominator dividing e ⇒ exp·e is an integer t-exponent.
        let k = (exp.clone() * Rational::from(e))
            .numer()
            .to_i64()
            .unwrap_or(0);
        ts.insert(k, c.clone());
    }
    ts
}

/// Series of `b = Σⱼ bⱼ(x) yʲ` along a branch `y = bts(t)`, `x = α + t^e`.
pub(super) fn elem_ts(b: &AlgElem, alpha: &Rational, e: i64, u: i64, bts: &TS) -> TS {
    let mut acc = TS::new();
    for (j, coeff) in b.iter().enumerate() {
        if coeff.numer().is_empty() {
            continue;
        }
        let cj = ratfn_ts(coeff, alpha, e, u);
        let yj = ts_pow(bts, j as u32, u);
        acc = ts_add(&acc, &ts_mul(&cj, &yj, u));
    }
    acc
}

fn ratfn_ts(r: &RatFn, alpha: &Rational, e: i64, u: i64) -> TS {
    let num = poly_ts(r.numer(), alpha, e, u);
    let den = poly_ts(r.denom(), alpha, e, u);
    match ts_inv(&den, u) {
        Some(inv) => ts_mul(&num, &inv, u),
        None => TS::new(),
    }
}

/// `p(α + t^e)` truncated to `t`-exponents `< u`.
fn poly_ts(p: &QPoly, alpha: &Rational, e: i64, u: i64) -> TS {
    let mut ts = TS::new();
    for (m, pm) in p.iter().enumerate() {
        if *pm == 0 {
            continue;
        }
        // (α + t^e)^m = Σ_l C(m,l) α^{m−l} t^{e l}.
        for l in 0..=m {
            let exp = e * l as i64;
            if exp >= u {
                break;
            }
            let coeff = pm.clone()
                * Rational::from(binom(m as u32, l as u32))
                * rat_pow(alpha, (m - l) as u32);
            if coeff != 0 {
                *ts.entry(exp).or_insert_with(|| Rational::from(0)) += &coeff;
            }
        }
    }
    ts.retain(|_, c| *c != 0);
    ts
}

pub(super) fn ts_add(a: &TS, b: &TS) -> TS {
    let mut r = a.clone();
    for (k, c) in b {
        *r.entry(*k).or_insert_with(|| Rational::from(0)) += c;
    }
    r.retain(|_, c| *c != 0);
    r
}

pub(super) fn ts_mul(a: &TS, b: &TS, u: i64) -> TS {
    let mut r = TS::new();
    for (ka, ca) in a {
        for (kb, cb) in b {
            let k = ka + kb;
            if k < u {
                *r.entry(k).or_insert_with(|| Rational::from(0)) += ca.clone() * cb;
            }
        }
    }
    r.retain(|_, c| *c != 0);
    r
}

pub(super) fn ts_pow(a: &TS, m: u32, u: i64) -> TS {
    let mut acc = TS::new();
    acc.insert(0, Rational::from(1));
    for _ in 0..m {
        acc = ts_mul(&acc, a, u);
    }
    acc
}

/// Inverse of a Laurent series, truncated to exponents `< u`.
pub(super) fn ts_inv(s: &TS, u: i64) -> Option<TS> {
    let (&v0, c0) = s.iter().next()?; // lowest exponent
    let inv_c0 = Rational::from(1) / c0.clone();
    let kmax = (u + v0).max(1);
    // Normalized u-series coefficients: u[k] = s[v0+k]/c0  (u[0] = 1).
    let mut us = vec![Rational::from(0); kmax as usize + 1];
    for (exp, c) in s {
        let k = exp - v0;
        if (0..=kmax).contains(&k) {
            us[k as usize] = c.clone() * &inv_c0;
        }
    }
    let mut iu = vec![Rational::from(0); kmax as usize + 1];
    iu[0] = Rational::from(1);
    for k in 1..=kmax as usize {
        let mut acc = Rational::from(0);
        for i in 1..=k {
            acc += us[i].clone() * &iu[k - i];
        }
        iu[k] = -acc;
    }
    let mut res = TS::new();
    for (k, iuk) in iu.iter().enumerate() {
        let exp = -v0 + k as i64;
        if exp < u && *iuk != 0 {
            res.insert(exp, inv_c0.clone() * iuk);
        }
    }
    Some(res)
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn to_monomials(f_coeffs: &[QPoly]) -> Vec<(u32, u32, Rational)> {
    let mut out = Vec::new();
    for (j, cj) in f_coeffs.iter().enumerate() {
        for (i, c) in cj.iter().enumerate() {
            if *c != 0 {
                out.push((i as u32, j as u32, c.clone()));
            }
        }
    }
    out
}

fn max_denom_degree(b: &[AlgElem], bd: &AlgElem) -> usize {
    let one = b
        .iter()
        .chain(std::iter::once(bd))
        .flat_map(|e| e.iter())
        .map(|c| degree(c.denom()).max(0) as usize)
        .max()
        .unwrap_or(0);
    one
}

fn scale(b: &AlgElem, s: &RatFn) -> AlgElem {
    let f = RationalFunctionField;
    b.iter().map(|c| f.mul(s, c)).collect()
}

fn binom(n: u32, k: u32) -> Integer {
    if k > n {
        return Integer::from(0);
    }
    let mut num = Integer::from(1);
    for t in 0..k {
        num *= Integer::from(n - t);
    }
    let mut den = Integer::from(1);
    for t in 1..=k {
        den *= Integer::from(t);
    }
    num / den
}

fn rat_pow(c: &Rational, e: u32) -> Rational {
    let mut acc = Rational::from(1);
    for _ in 0..e {
        acc *= c;
    }
    acc
}

/// Solve `M·a = rhs` over `ℚ` for `ncols` unknowns; particular solution (free
/// vars 0) or `None` if inconsistent.
fn gauss_solve(
    mut m: Vec<Vec<Rational>>,
    mut b: Vec<Rational>,
    ncols: usize,
) -> Option<Vec<Rational>> {
    let nrows = m.len();
    let mut pivot_of_col = vec![None; ncols];
    let mut row = 0usize;
    for col in 0..ncols {
        if row >= nrows {
            break;
        }
        let Some(sel) = (row..nrows).find(|&r| m[r][col] != 0) else {
            continue;
        };
        m.swap(row, sel);
        b.swap(row, sel);
        let piv = m[row][col].clone();
        for v in m[row].iter_mut() {
            *v /= &piv;
        }
        b[row] /= &piv;
        let pr = m[row].clone();
        let pb = b[row].clone();
        for r in 0..nrows {
            if r != row && m[r][col] != 0 {
                let f = m[r][col].clone();
                for (dst, pv) in m[r].iter_mut().zip(pr.iter()) {
                    *dst -= f.clone() * pv;
                }
                b[r] -= f * &pb;
            }
        }
        pivot_of_col[col] = Some(row);
        row += 1;
    }
    for r in 0..nrows {
        if m[r].iter().all(|v| *v == 0) && b[r] != 0 {
            return None; // inconsistent
        }
    }
    let mut x = vec![Rational::from(0); ncols];
    for (col, pr) in pivot_of_col.iter().enumerate() {
        if let Some(r) = pr {
            x[col] = b[*r].clone();
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }

    /// Power basis for a non-singular curve: y² − x − 1 (disc = −4, no repeated
    /// roots) ⇒ {1, y}.
    #[test]
    fn nonsingular_power_basis() {
        let basis = integral_basis(&[qp(&[-1, -1]), qp(&[]), qp(&[1])]).expect("basis");
        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0], vec![RatFn::int(1)]);
        assert_eq!(basis[1], vec![RatFn::int(0), RatFn::int(1)]);
    }

    /// Cusp y² = x³ ⇒ {1, y/x}: a genuine enlargement at the singular x=0.
    #[test]
    fn cusp_basis() {
        let basis = integral_basis(&[qp(&[0, 0, 0, -1]), qp(&[]), qp(&[1])]).expect("basis");
        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0], vec![RatFn::int(1)]);
        // b₁ = y/x.
        assert_eq!(
            basis[1],
            vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[0, 1]))]
        );
        // All integral.
        let f = [qp(&[0, 0, 0, -1]), qp(&[]), qp(&[1])];
        assert!(basis.iter().all(|bi| is_integral(&f, bi)));
    }

    /// Nodal cubic y² = x³ + x² (non-radical) ⇒ {1, y/x}; (y/x)² = x + 1.
    #[test]
    fn nodal_cubic_basis() {
        // F = y² − x² − x³.
        let f = [qp(&[0, 0, -1, -1]), qp(&[]), qp(&[1])];
        let basis = integral_basis(&f).expect("basis");
        assert_eq!(basis.len(), 2);
        assert_eq!(
            basis[1],
            vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[0, 1]))]
        );
        assert!(basis.iter().all(|bi| is_integral(&f, bi)));
    }

    /// Degree-3 radical y³ = x² ⇒ {1, y, y²/x}.
    #[test]
    fn cubic_radical_basis() {
        let f = [qp(&[0, 0, 1]), qp(&[]), qp(&[]), qp(&[1])]; // y³ − x²
        let basis = integral_basis(&f).expect("basis");
        assert_eq!(basis.len(), 3);
        assert_eq!(basis[1], vec![RatFn::int(0), RatFn::int(1)]); // y
        assert_eq!(
            basis[2],
            vec![
                RatFn::int(0),
                RatFn::int(0),
                RatFn::new(qp(&[1]), qp(&[0, 1]))
            ]
        ); // y²/x
        assert!(basis.iter().all(|bi| is_integral(&f, bi)));
    }
}
