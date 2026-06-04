//! Integral bases of algebraic function fields `ℚ(x)(y)`, `F(x,y)=0` — Risch
//! milestone **MB** (van Hoeij 1994) substrate.
//!
//! An *integral basis* `w₀, …, w_{n−1}` of `ℚ(x)(y)` over `ℚ[x]` spans the
//! integral closure of `ℚ[x]` — exactly the functions with **no finite poles**.
//! It is the normalizing model Trager's algebraic-integration algorithm (M3)
//! builds Hermite reduction and residue analysis on.
//!
//! This module provides the verifiable, Puiseux-free **core primitives** of the
//! computation:
//!
//! * [`discriminant`] — `D(x) = det[Tr(yⁱ⁺ʲ)] = Res_y(F, ∂F/∂y)` (up to sign),
//!   computed from the trace form; its repeated factors locate the singular
//!   places.
//! * [`rational_singularities`] — the rational `α` with `(x−α)² | D`, the only
//!   finite places where the naïve basis `1, y, …, y^{n−1}` can fail to be
//!   integral.
//! * [`is_integral`] — an **exact** integrality test: `a` is integral over `ℚ[x]`
//!   iff its characteristic polynomial (over `ℚ(x)`) has polynomial coefficients,
//!   computed from the traces `Tr(aᵏ)` via Newton's identities.  Sound and
//!   complete — no truncation, no Puiseux precision bound.
//! * [`radical_integral_basis`] — Trager's explicit basis `wᵢ = yⁱ/dᵢ` for a
//!   simple radical `yⁿ = p(x)` (the case van Hoeij/Trager note is *immediate*),
//!   each element verified by [`is_integral`].
//!
//! The general (non-radical) van Hoeij enlargement loop — which uses Puiseux
//! expansions at each singular place to solve a linear system for the next basis
//! element — is the documented next step; it needs Puiseux with algebraic
//! coefficients at algebraic singular places (see `poly::puiseux`).

use rug::Rational;

use super::super::risch::alg_field::{AlgElem, AlgExtension, RatFn, RationalFunctionField};
use super::super::risch::number_field::CoeffField;
use super::super::risch::poly_rde::{degree, poly_deriv, poly_mul, trim, QPoly};
use super::super::risch::rational_rde::{poly_div_exact, poly_gcd};

/// `Tr_{L/ℚ(x)}(b)` for `b ∈ ℚ(x)(y)`: the trace of multiplication-by-`b`,
/// `Σⱼ [b·yʲ]ⱼ` (the `yʲ`-coefficient of `b·yʲ` reduced mod `F`).
fn trace(e: &AlgExtension, b: &AlgElem) -> RatFn {
    let f = RationalFunctionField;
    let n = e.degree() as usize;
    let mut acc = f.zero();
    let gen = e.generator();
    let mut yj = e.from_int(1); // y^0
    for j in 0..n {
        let prod = e.mul(b, &yj);
        if let Some(c) = prod.get(j) {
            acc = f.add(&acc, c);
        }
        if j + 1 < n {
            yj = e.mul(&yj, &gen);
        }
    }
    acc
}

/// The discriminant `D(x) = det[Tr(yⁱ⁺ʲ)]_{0≤i,j<n}` of the power basis — equal
/// to `Res_y(F, ∂F/∂y)` up to sign.  A `ℚ[x]` polynomial (the basis is integral).
pub fn discriminant(f_coeffs: &[QPoly]) -> QPoly {
    let e = AlgExtension::new(f_coeffs);
    let n = e.degree() as usize;
    let gen = e.generator();
    // Powers y^0 … y^{2n−2} and their traces.
    let ntr = 2 * n.saturating_sub(1) + 1;
    let mut tr = vec![RatFn::int(0); ntr];
    let mut ym = e.from_int(1);
    for (m, slot) in tr.iter_mut().enumerate() {
        *slot = trace(&e, &ym);
        if m + 1 < ntr {
            ym = e.mul(&ym, &gen);
        }
    }
    // Matrix M[i][j] = Tr(y^{i+j}); det over ℚ(x).
    let mat: Vec<Vec<RatFn>> = (0..n)
        .map(|i| (0..n).map(|j| tr[i + j].clone()).collect())
        .collect();
    let det = ratfn_det(mat);
    // det ∈ ℚ[x] (denominator 1); return its numerator.
    debug_assert!(is_polynomial(&det));
    trim(det.numer().clone())
}

/// The rational singular places: `α ∈ ℚ` with `(x−α)² | D` (i.e. `α` is a
/// repeated root of the discriminant).  These are the only finite places where
/// the naïve basis can fail integrality.  (Algebraic singular places exist too;
/// they are out of this rational-only scope.)
pub fn rational_singularities(disc: &QPoly) -> Vec<Rational> {
    let disc = trim(disc.clone());
    if degree(&disc) < 2 {
        return Vec::new();
    }
    // Repeated part = gcd(D, D′); its rational roots are the repeated roots of D.
    let g = poly_gcd(&disc, &poly_deriv(&disc));
    rational_roots_monic(&g)
}

/// Decide whether `a ∈ ℚ(x)(y)` (given as the `yʲ`-coefficient vector over `ℚ(x)`)
/// is integral over `ℚ[x]`.  Exact: `a` is integral iff its characteristic
/// polynomial `Tⁿ − e₁Tⁿ⁻¹ + … ± eₙ` over `ℚ(x)` has every `eₖ ∈ ℚ[x]`.  The power
/// sums `pₖ = Tr(aᵏ)` give the `eₖ` by Newton's identities.
pub fn is_integral(f_coeffs: &[QPoly], a: &AlgElem) -> bool {
    let e = AlgExtension::new(f_coeffs);
    let f = RationalFunctionField;
    let n = e.degree() as usize;
    // Power sums p_k = Tr(a^k), k = 1..=n.
    let mut p = vec![f.zero(); n + 1];
    let mut ak = e.from_int(1);
    for pk in p.iter_mut().take(n + 1).skip(1) {
        ak = e.mul(&ak, a);
        *pk = trace(&e, &ak);
    }
    // Newton: k·e_k = Σ_{i=1}^k (−1)^{i−1} e_{k−i} p_i,  e_0 = 1.
    let mut ek = vec![f.zero(); n + 1];
    ek[0] = f.one();
    for k in 1..=n {
        let mut acc = f.zero();
        for i in 1..=k {
            let term = f.mul(&ek[k - i], &p[i]);
            if i % 2 == 1 {
                acc = f.add(&acc, &term);
            } else {
                acc = f.add(&acc, &f.neg(&term));
            }
        }
        // e_k = acc / k
        let inv_k = f
            .inv(&RatFn::int(k as i64))
            .expect("k≠0 is invertible in ℚ(x)");
        ek[k] = f.mul(&acc, &inv_k);
    }
    // Integral iff every e_k is a polynomial (denominator 1).
    ek.iter().skip(1).all(is_polynomial)
}

/// Trager's explicit integral basis for a **simple radical** `yⁿ = p(x)`:
/// writing `p = ∏ⱼ pⱼʲ` (squarefree form, `pⱼ` pairwise coprime, squarefree),
/// `wᵢ = yⁱ / dᵢ` with `dᵢ = ∏ⱼ pⱼ^{⌊i·j/n⌋}`.  Returns the basis as `AlgElem`s
/// over `ℚ[x]/(yⁿ − p)`.  Each element is checked by [`is_integral`].
pub fn radical_integral_basis(n: usize, p: &QPoly) -> Option<Vec<AlgElem>> {
    if n < 2 {
        return None;
    }
    // F = yⁿ − p.
    let mut f_coeffs = vec![QPoly::new(); n + 1];
    f_coeffs[0] = trim(poly_scale(p, &Rational::from(-1)));
    f_coeffs[n] = vec![Rational::from(1)];

    // Squarefree decomposition p = ∏ pⱼ^j  (Yun).
    let sqfree = squarefree_factors(p);

    let mut basis = Vec::with_capacity(n);
    for i in 0..n {
        // dᵢ = ∏ⱼ pⱼ^{⌊i·j/n⌋}.
        let mut di = vec![Rational::from(1)];
        for (j, pj) in sqfree.iter().enumerate() {
            let mult = (i * (j + 1)) / n; // pⱼ has multiplicity (j+1)
            for _ in 0..mult {
                di = poly_mul(&di, pj);
            }
        }
        // wᵢ = (1/dᵢ)·yⁱ : AlgElem with component i = 1/dᵢ.
        let mut w = vec![RatFn::int(0); i + 1];
        w[i] = RatFn::new(vec![Rational::from(1)], di);
        if !is_integral(&f_coeffs, &w) {
            return None; // safety: the formula must yield integral elements
        }
        basis.push(w);
    }
    Some(basis)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_polynomial(r: &RatFn) -> bool {
    let d = r.denom();
    r.numer().is_empty() || (d.len() == 1 && d[0] == 1)
}

fn poly_scale(p: &QPoly, s: &Rational) -> QPoly {
    p.iter().map(|c| c.clone() * s).collect()
}

/// Determinant of a square matrix over `ℚ(x)` by Gaussian elimination.
fn ratfn_det(mut m: Vec<Vec<RatFn>>) -> RatFn {
    let f = RationalFunctionField;
    let n = m.len();
    let mut det = f.one();
    for col in 0..n {
        // Find a pivot.
        let Some(piv) = (col..n).find(|&r| !is_zero(&m[r][col])) else {
            return f.zero();
        };
        if piv != col {
            m.swap(piv, col);
            det = f.neg(&det);
        }
        det = f.mul(&det, &m[col][col]);
        let inv = f.inv(&m[col][col]).expect("nonzero pivot invertible");
        for r in (col + 1)..n {
            if is_zero(&m[r][col]) {
                continue;
            }
            let factor = f.mul(&m[r][col], &inv);
            let pivot_row = m[col].clone();
            for (c, pv) in pivot_row.iter().enumerate().skip(col) {
                let sub = f.mul(&factor, pv);
                m[r][c] = f.add(&m[r][c], &f.neg(&sub));
            }
        }
    }
    det
}

fn is_zero(r: &RatFn) -> bool {
    r.numer().is_empty()
}

/// Yun squarefree factorization of `p ∈ ℚ[x]`: returns `[p₁, p₂, …]` with
/// `p = ∏ⱼ pⱼ^{j+1}` (each `pⱼ` squarefree, pairwise coprime; `pⱼ` may be `1`).
fn squarefree_factors(p: &QPoly) -> Vec<QPoly> {
    let p = trim(p.clone());
    if degree(&p) < 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let dp = poly_deriv(&p);
    let mut a = poly_gcd(&p, &dp); // ∏ pⱼ^{j}
    let mut b = poly_div_exact(&p, &a); // ∏ pⱼ
    loop {
        let c = poly_gcd(&a, &b); // ∏_{j≥current} pⱼ
        let factor = poly_div_exact(&b, &c); // current squarefree factor
        out.push(trim(factor));
        if degree(&a) < 1 {
            break;
        }
        a = poly_div_exact(&a, &c);
        b = c;
        if degree(&b) < 1 {
            break;
        }
    }
    out
}

/// Distinct rational roots of a monic-ish `ℚ[x]` polynomial via the rational
/// root theorem.
fn rational_roots_monic(p: &QPoly) -> Vec<Rational> {
    use rug::Integer;
    let p = trim(p.clone());
    if degree(&p) < 1 {
        return Vec::new();
    }
    let mut den_lcm = Integer::from(1);
    for c in &p {
        den_lcm = den_lcm.lcm(c.denom());
    }
    let ints: Vec<Integer> = p
        .iter()
        .map(|c| {
            (c.clone() * Rational::from(den_lcm.clone()))
                .numer()
                .clone()
        })
        .collect();
    // Trim leading zeros (shouldn't be any after trim) and trailing.
    let a0 = ints[0].clone().abs();
    let an = ints[ints.len() - 1].clone().abs();
    let mut roots = Vec::new();
    if ints[0] == 0 {
        roots.push(Rational::from(0));
    }
    let pdiv = divisors(&a0);
    let qdiv = divisors(&an);
    for pn in &pdiv {
        for qn in &qdiv {
            for sign in [1i32, -1] {
                let cand = Rational::from((Integer::from(sign) * pn.clone(), qn.clone()));
                if roots.contains(&cand) {
                    continue;
                }
                let mut acc = Rational::from(0);
                for a in ints.iter().rev() {
                    acc = acc * &cand + Rational::from(a.clone());
                }
                if acc == 0 {
                    roots.push(cand);
                }
            }
        }
    }
    roots
}

fn divisors(n: &rug::Integer) -> Vec<rug::Integer> {
    use rug::Integer;
    let n = n.clone().abs();
    if n == 0 {
        return vec![Integer::from(1)];
    }
    let mut ds = Vec::new();
    let mut d = Integer::from(1);
    while Integer::from(&d * &d) <= n {
        if n.is_divisible(&d) {
            ds.push(d.clone());
            let o = n.clone() / &d;
            if o != d {
                ds.push(o);
            }
        }
        d += 1;
    }
    ds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }
    fn r(n: i64) -> Rational {
        Rational::from(n)
    }

    #[test]
    fn discriminant_sqrt() {
        // F = y² − x ⇒ disc = ±4x.
        let d = discriminant(&[qp(&[0, -1]), qp(&[]), qp(&[1])]);
        // 4x (sign aside): degree 1, root at 0 simple.
        assert_eq!(degree(&d), 1);
        assert!(
            rational_singularities(&d).is_empty(),
            "x=0 is a simple root"
        );
    }

    #[test]
    fn discriminant_cusp_singular() {
        // F = y² − x³ ⇒ disc = ±4x³, repeated root at x=0.
        let d = discriminant(&[qp(&[0, 0, 0, -1]), qp(&[]), qp(&[1])]);
        assert_eq!(degree(&d), 3);
        let s = rational_singularities(&d);
        assert_eq!(s, vec![r(0)]);
    }

    #[test]
    fn is_integral_examples() {
        // F = y² − x³.
        let f = [qp(&[0, 0, 0, -1]), qp(&[]), qp(&[1])];
        // y is integral.
        assert!(is_integral(&f, &vec![RatFn::int(0), RatFn::int(1)]));
        // y/x is integral ((y/x)² = x).
        let y_over_x = vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[0, 1]))];
        assert!(is_integral(&f, &y_over_x));
        // y/x² is NOT integral ((y/x²)² = 1/x).
        let y_over_x2 = vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[0, 0, 1]))];
        assert!(!is_integral(&f, &y_over_x2));
    }

    #[test]
    fn radical_basis_sqrt_x_is_power_basis() {
        // y² − x: radicand x squarefree ⇒ d₀=d₁=1, basis {1, y}.
        let basis = radical_integral_basis(2, &qp(&[0, 1])).expect("basis");
        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0], vec![RatFn::int(1)]);
        assert_eq!(basis[1], vec![RatFn::int(0), RatFn::int(1)]);
    }

    #[test]
    fn radical_basis_y2_eq_x3() {
        // y² = x³ : p = x³ = x²·x, squarefree factors [x, ... ]; d₁ = ⌊1·?⌋.
        // x³ has squarefree form p₃ part; for n=2, d₁ should give y/x integral.
        let basis = radical_integral_basis(2, &qp(&[0, 0, 0, 1])).expect("basis");
        assert_eq!(basis.len(), 2);
        // w₁ must be integral and equal y/x.
        assert_eq!(
            basis[1],
            vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[0, 1]))]
        );
    }

    #[test]
    fn radical_basis_cbrt_x_squared() {
        // y³ = x² : the eq-11 basis is {1, y, y²/x}.  All verified integral.
        let basis = radical_integral_basis(3, &qp(&[0, 0, 1])).expect("basis");
        assert_eq!(basis.len(), 3);
        // w₂ = y²/x.
        assert_eq!(
            basis[2],
            vec![
                RatFn::int(0),
                RatFn::int(0),
                RatFn::new(qp(&[1]), qp(&[0, 1]))
            ]
        );
    }
}
