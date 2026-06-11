//! K-rational integration **with logarithms** — Risch §E follow-up.
//!
//! [`super::rational_rde::solve_rational_rde_k`] decides whether `∫ c(x) dx`
//! (`c ∈ K(x)`, `K = ℚ(α)` a number field) has a **K-rational** antiderivative,
//! i.e. one with no new transcendental generators.  When it returns `None` the
//! antiderivative may still be **elementary** — it just needs new `log` terms
//! with `K`-coefficients, e.g.
//!
//! ```text
//!   ∫ 1/(x·(x+√2)) dx = (1/√2)·[log(x) − log(x+√2)]
//! ```
//!
//! This module provides `integrate_k_rational_with_logs`, a Rothstein–Trager
//! / partial-fractions step over `K` that produces the **rational part** plus a
//! list of `(residue, linear factor)` pairs `(cᵢ, x − rᵢ)` with `cᵢ, rᵢ ∈ K`,
//! i.e. `Σ cᵢ·log(x − rᵢ)`.
//!
//! ## Scope
//!
//! - The polynomial part `Q` of `c_num = Q·c_den + R` is always integrated
//!   (`∫Q` is a `K`-polynomial — always elementary).
//! - The proper fraction `R/D` (`D` made monic, `gcd(R,D)=1`) must have a
//!   **squarefree** denominator `D`.  Repeated factors (Hermite reduction) are
//!   **not yet handled** — declines (`None`).  (The Hermite-reduced rational
//!   part can still be produced separately by
//!   [`super::rational_rde::solve_rational_rde_k`] when no logs are needed for
//!   that piece; combining the two is a follow-up.)
//! - `D` must split **completely into distinct `K`-linear factors** (`deg ≤ 2`
//!   with the quadratic case requiring its discriminant to be a perfect square
//!   in `K`, currently only for `K = ℚ(√d)` — `AlgebraicExtension::SingleSqrt`).
//!   For each root `rᵢ ∈ K` of `D`, the residue is the simple-pole formula
//!   `cᵢ = R(rᵢ) / D'(rᵢ)` evaluated by `K`-arithmetic.
//! - Non-linear irreducible `K`-factors of `D` (e.g. an irreducible quadratic
//!   over `K` whose discriminant is not a `K`-square) **decline** (`None`).
//!   Documented as a follow-up: would need a `K`-algebraic extension `K(β)`
//!   (Lazard–Rioboo–Trager `RootSum`-style) to express the residues.

use rug::Rational;

use super::exp_case::{rational_sqrt, AlgebraicExtension};
use super::number_field::{KElem, KPoly, NumberField};

/// Already-integrated Hermite rational terms `(B, V^p, p)`, contributing
/// `B/V^p` to the antiderivative.
type KHermiteTerms = Vec<(KPoly, KPoly, usize)>;

/// Result of `integrate_k_rational_with_logs`:
///
/// - `poly_part`: a `K`-polynomial `Q` whose `∫Q dx` is the polynomial piece
///   of the antiderivative (the caller integrates it term-by-term).
/// - `hermite_terms`: `(B, V, p)` triples — **already-integrated** rational
///   pieces `B/V^p` (`p ≥ 1`) produced by Hermite reduction of repeated `K`-
///   factors of the denominator.  These are *not* integrated further.
/// - `log_terms`: `(residue, root)` pairs contributing `Σ residue·log(x −
///   root)` for the squarefree logarithmic remainder.
pub struct KRationalLogResult {
    /// `K`-polynomial part `Q`: `∫Q dx` is elementary (caller integrates).
    pub poly_part: KPoly,
    /// Already-integrated Hermite rational terms `B/V^p`.
    pub hermite_terms: KHermiteTerms,
    /// `(residue, root)` pairs: contributes `Σ residue·log(x − root)`.
    pub log_terms: Vec<(KElem, KElem)>,
}

/// Evaluate a `K`-polynomial `p(x)` at `x = point ∈ K` via Horner's method.
fn eval_kpoly_at(field: &NumberField, p: &[KElem], point: &KElem) -> KElem {
    let mut acc = NumberField::k_zero();
    for c in p.iter().rev() {
        acc = field.add(&field.mul(&acc, point), c);
    }
    acc
}

/// Square root of a `K`-element `D = a + b·√d` in `K = ℚ(√d)` (modulus
/// `t² − d`), or `None` if `D` is not a perfect square in `K`.
///
/// Solves `(p + q√d)² = D`, i.e. `p² + d·q² = a` and `2pq = b`:
/// - `b = 0`: either `q = 0, p = √a` or `p = 0, q = √(a/d)`.
/// - `b ≠ 0`: `p² = (a ± √(a² − d·b²)) / 2` for some sign, with `q = b/(2p)`.
fn k_sqrt_quadratic(field: &NumberField, d: i64, elem: &KElem) -> Option<KElem> {
    let zero = Rational::from(0);
    let a = elem.first().cloned().unwrap_or_else(|| zero.clone());
    let b = elem.get(1).cloned().unwrap_or_else(|| zero.clone());
    let d_r = Rational::from(d);

    if b == 0 {
        if let Some(p) = rational_sqrt(&a) {
            return Some(field.from_rational(&p));
        }
        // q = sqrt(a/d)
        if a != 0 {
            let aod = a.clone() / d_r.clone();
            if let Some(q) = rational_sqrt(&aod) {
                return Some(vec![zero, q]);
            }
        }
        return None;
    }

    // a² − d·b²
    let disc = a.clone() * a.clone() - d_r.clone() * b.clone() * b.clone();
    let s = rational_sqrt(&disc)?;
    for sign in [1i32, -1i32] {
        let numer = if sign == 1 {
            a.clone() + s.clone()
        } else {
            a.clone() - s.clone()
        };
        if numer < 0 {
            continue;
        }
        let p_sq = numer / Rational::from(2);
        if let Some(p) = rational_sqrt(&p_sq) {
            if p == 0 {
                continue; // p = 0 would require b = 0, handled above.
            }
            let q = b.clone() / (Rational::from(2) * p.clone());
            return Some(vec![p, q]);
        }
    }
    None
}

/// Find all roots of a monic `K`-polynomial `den` of `K`-degree 1 or 2 in `K`.
///
/// Returns `None` if `den` has degree ≥ 3, or (for degree 2) its discriminant
/// is not a `K`-square — i.e. `den` does not split completely into distinct
/// `K`-linear factors.  `ext` selects the field-specific square-root routine
/// (currently only `AlgebraicExtension::SingleSqrt` supports the degree-2
/// case).
fn find_k_roots(
    field: &NumberField,
    den: &[KElem],
    ext: &AlgebraicExtension,
) -> Option<Vec<KElem>> {
    match NumberField::kdeg(den) {
        1 => {
            // den = x + c  (monic)  ⇒  root = −c.
            let c = den.first().cloned().unwrap_or_else(NumberField::k_zero);
            Some(vec![field.neg(&c)])
        }
        2 => {
            // den = x² + b x + c (monic).  Discriminant Δ = b² − 4c.
            let b = den.get(1).cloned().unwrap_or_else(NumberField::k_zero);
            let c = den.first().cloned().unwrap_or_else(NumberField::k_zero);
            let four_c = field.mul(&c, &field.from_int(4));
            let disc = field.sub(&field.mul(&b, &b), &four_c);
            let AlgebraicExtension::SingleSqrt { d, .. } = ext else {
                return None; // K-sqrt not implemented for this extension shape
            };
            let sqrt_disc = k_sqrt_quadratic(field, *d, &disc)?;
            let two = field.from_int(2);
            let two_inv = field.inv(&two)?;
            let neg_b = field.neg(&b);
            let r1 = field.mul(&field.add(&neg_b, &sqrt_disc), &two_inv);
            let r2 = field.mul(&field.sub(&neg_b, &sqrt_disc), &two_inv);
            if NumberField::kpoly_eq(std::slice::from_ref(&r1), std::slice::from_ref(&r2)) {
                return None; // repeated root: den not squarefree (shouldn't happen)
            }
            Some(vec![r1, r2])
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Hermite reduction over K (Bronstein §2.2 / Yun, K-coefficient version)
// ---------------------------------------------------------------------------

/// Yun squarefree factorization of a monic `K`-polynomial: returns `(Vᵢ, i)`
/// for each non-constant `Vᵢ`, with `f = ∏ Vᵢ^i` and the `Vᵢ` squarefree &
/// pairwise coprime.  `K`-analogue of [`super::rational_integrate::yun`].
///
/// Returns `None` if a leading-coefficient inversion fails (shouldn't happen
/// for `f` already monic over a genuine field) or the iteration would not
/// terminate (safety bound, characteristic 0 — should never trigger).
fn kyun(field: &NumberField, f: &KPoly) -> Option<Vec<(KPoly, usize)>> {
    let f = field.kpoly_monic(f)?;
    if NumberField::kdeg(&f) <= 0 {
        return Some(vec![]);
    }
    let fp = field.kpoly_deriv(&f);
    let a0 = field.kpoly_gcd(&f, &fp)?;
    if NumberField::kdeg(&a0) == 0 {
        return Some(vec![(f, 1)]); // already squarefree
    }
    let mut b = field.kpoly_div_exact(&f, &a0)?;
    let c = field.kpoly_div_exact(&fp, &a0)?;
    let mut d = field.kpoly_sub(&c, &field.kpoly_deriv(&b));
    let mut result = Vec::new();
    let mut i = 1usize;
    let cap = NumberField::kdeg(&f) as usize + 2;
    while NumberField::kdeg(&b) > 0 {
        if i > cap {
            return None; // safety: should never trigger in characteristic 0
        }
        let vi = field.kpoly_gcd(&b, &d)?;
        if NumberField::kdeg(&vi) > 0 {
            result.push((field.kpoly_monic(&vi)?, i));
        }
        let b_next = field.kpoly_div_exact(&b, &vi)?;
        let c_next = field.kpoly_div_exact(&d, &vi)?;
        d = field.kpoly_sub(&c_next, &field.kpoly_deriv(&b_next));
        b = b_next;
        i += 1;
    }
    Some(result)
}

/// Partial-fraction decomposition over pairwise-coprime `K`-moduli: returns
/// `Aᵢ` with `num/∏mᵢ = Σ Aᵢ/mᵢ` and `deg_x Aᵢ < deg_x mᵢ`.  `K`-analogue of
/// `super::rational_integrate::partial_fractions`.
fn kpartial_fractions(field: &NumberField, num: &KPoly, moduli: &[KPoly]) -> Option<Vec<KPoly>> {
    let n = moduli.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return field.kpoly_mod(num, &moduli[0]).map(|a| vec![a]);
    }
    let one = vec![field.from_int(1)];
    let mut result = Vec::with_capacity(n);
    let mut cur = NumberField::kpoly_trim(num.clone());
    for i in 0..n - 1 {
        let mi = &moduli[i];
        let rest = moduli[i + 1..]
            .iter()
            .fold(one.clone(), |acc, m| field.kpoly_mul(&acc, m));
        let (g, _s, t) = field.kpoly_ext_gcd(mi, &rest)?;
        if NumberField::kdeg(&g) != 0 {
            return None; // moduli not coprime
        }
        let ai = field.kpoly_mod(&field.kpoly_mul(&cur, &t), mi)?;
        let s = field.kpoly_div_exact(&field.kpoly_sub(&cur, &field.kpoly_mul(&ai, &rest)), mi)?;
        result.push(ai);
        cur = s;
    }
    result.push(cur);
    Some(result)
}

/// Hermite reduction of `aᵢ / V^k` (`V` squarefree over `K`, surrounding
/// cofactor `1`).  `K`-analogue of `super::rational_integrate::hermite_factor`.
///
/// Returns the rational-part terms `(B, p)` (contributing `B/V^p`) and the
/// leftover numerator over `V^1` (feeding the squarefree logarithmic part).
fn khermite_factor(
    field: &NumberField,
    ai: &KPoly,
    v: &KPoly,
    k: usize,
) -> Option<(Vec<(KPoly, usize)>, KPoly)> {
    let vp = field.kpoly_deriv(v);
    let mut a = NumberField::kpoly_trim(ai.clone());
    let mut terms = Vec::new();
    let mut power = k;
    while power >= 2 {
        let factor = field.from_int((power - 1) as i64);
        let coeff = field.kpoly_mod(&field.kpoly_scale(&vp, &factor), v)?;
        let inv = field.kpoly_mod_inverse(&coeff, v)?;
        let neg_one = field.from_int(-1);
        let b = field.kpoly_mod(&field.kpoly_mul(&field.kpoly_scale(&a, &neg_one), &inv), v)?;
        // numerator = a − (B'·V − (k−1)·V'·B)
        let inner = field.kpoly_sub(
            &field.kpoly_mul(&field.kpoly_deriv(&b), v),
            &field.kpoly_scale(&field.kpoly_mul(&vp, &b), &factor),
        );
        a = field.kpoly_div_exact(&field.kpoly_sub(&a, &inner), v)?;
        terms.push((b, power - 1));
        power -= 1;
    }
    Some((terms, a))
}

/// Full Hermite reduction of a proper `K`-fraction `r/d` (`d` monic,
/// `gcd_x(r,d)=1`).  `K`-analogue of `super::rational_integrate::hermite_reduce`.
///
/// Returns `(hermite_terms, h, drad)`: the already-integrated `B/V^p` pieces,
/// and the proper fraction `H/Drad` (`Drad` squarefree) feeding the
/// logarithmic part.
fn khermite_reduce(
    field: &NumberField,
    r: &KPoly,
    d: &KPoly,
) -> Option<(KHermiteTerms, KPoly, KPoly)> {
    let sqf = kyun(field, d)?;
    if sqf.is_empty() {
        return Some((vec![], r.clone(), d.clone()));
    }
    let one = vec![field.from_int(1)];
    let moduli: Vec<KPoly> = sqf
        .iter()
        .map(|(v, i)| {
            let mut m = one.clone();
            for _ in 0..*i {
                m = field.kpoly_mul(&m, v);
            }
            m
        })
        .collect();
    let parts = kpartial_fractions(field, r, &moduli)?;

    let drad: KPoly = sqf
        .iter()
        .fold(one.clone(), |acc, (v, _)| field.kpoly_mul(&acc, v));
    let mut hermite_terms: KHermiteTerms = Vec::new();
    let mut h = Vec::new();

    for ((v, i), ai) in sqf.iter().zip(parts.iter()) {
        let cofactor = field.kpoly_div_exact(&drad, v)?; // Drad / V_i
        if *i == 1 {
            // Already squarefree: the whole part feeds the log part.
            h = field.kpoly_add(&h, &field.kpoly_mul(ai, &cofactor));
            continue;
        }
        let (terms, leftover) = khermite_factor(field, ai, v, *i)?;
        for (b, p) in terms {
            if NumberField::kdeg(&b) < 0 {
                continue; // zero numerator — no contribution
            }
            let v_pow = field.kpoly_pow(v, p as u32);
            hermite_terms.push((b, v_pow, p));
        }
        h = field.kpoly_add(&h, &field.kpoly_mul(&leftover, &cofactor));
    }

    Some((hermite_terms, NumberField::kpoly_trim(h), drad))
}

/// Try to compute `∫ c_num/c_den dx` over `K = ℚ(α)`, allowing `log` terms
/// with `K`-coefficients in the result.
///
/// `c_num`, `c_den` are `K`-polynomials in `x`; `field`/`ext` describe `K`.
///
/// Returns `None` when:
/// - `c_den` reduces to a polynomial (no logs needed — caller should use the
///   plain `solve_rational_rde_k` path instead),
/// - the squarefree remainder's denominator does not split completely into
///   distinct `K`-linear factors (non-linear irreducible `K`-factor) — even
///   after Hermite reduction has peeled off any repeated factors.
pub(super) fn integrate_k_rational_with_logs(
    field: &NumberField,
    ext: &AlgebraicExtension,
    c_num: &KPoly,
    c_den: &KPoly,
) -> Option<KRationalLogResult> {
    let c_num = NumberField::kpoly_trim(c_num.clone());
    let c_den = NumberField::kpoly_trim(c_den.clone());
    if c_den.is_empty() {
        return None; // division by zero
    }
    if NumberField::kdeg(&c_den) < 1 {
        return None; // polynomial integrand — no logs needed
    }

    // Reduce to lowest terms with c_den monic.
    let g = field.kpoly_gcd(&c_num, &c_den)?;
    let num = field.kpoly_div_exact(&c_num, &g)?;
    let den_raw = field.kpoly_div_exact(&c_den, &g)?;
    let den = field.kpoly_monic(&den_raw)?;
    let lead_inv = {
        let lead = den_raw[NumberField::kdeg(&den_raw) as usize].clone();
        field.inv(&lead)?
    };
    let num = field.kpoly_scale(&num, &lead_inv);

    if NumberField::kdeg(&den) < 1 {
        return None; // reduced to a polynomial
    }

    // Polynomial division: num = q·den + r.
    let (q, r) = field.kpoly_divrem(&num, &den)?;
    let r = NumberField::kpoly_trim(r);

    if r.is_empty() {
        // Exact division — purely polynomial, handled by the caller's
        // K-rational path; no logs needed.
        return None;
    }

    // Hermite reduction: peel repeated K-factors of `den` into `hermite_terms`,
    // leaving a proper fraction H/Drad with Drad squarefree over K.
    let (hermite_terms, h, drad) = khermite_reduce(field, &r, &den)?;

    let mut log_terms = Vec::new();
    let h = NumberField::kpoly_trim(h);
    if !h.is_empty() && NumberField::kdeg(&drad) >= 1 {
        // Reduce H/Drad to lowest terms (Hermite leaves gcd(H,Drad)=1 in
        // exact arithmetic, but re-check defensively).
        let gh = field.kpoly_gcd(&h, &drad)?;
        let h = field.kpoly_div_exact(&h, &gh)?;
        let drad = field.kpoly_monic(&field.kpoly_div_exact(&drad, &gh)?)?;

        if NumberField::kdeg(&drad) >= 1 {
            let dprime = field.kpoly_deriv(&drad);

            // Find all roots of `drad` in K (must split completely).
            let roots = find_k_roots(field, &drad, ext)?;
            if roots.len() != NumberField::kdeg(&drad) as usize {
                return None; // non-linear irreducible K-factor
            }

            // Simple-pole residues: cᵢ = H(rᵢ) / Drad'(rᵢ).
            for root in &roots {
                let h_val = eval_kpoly_at(field, &h, root);
                let dp_val = eval_kpoly_at(field, &dprime, root);
                let dp_inv = field.inv(&dp_val)?;
                let residue = field.mul(&h_val, &dp_inv);
                if NumberField::is_zero(&residue) {
                    continue;
                }
                log_terms.push((residue, root.clone()));
            }
        }
    }

    Some(KRationalLogResult {
        poly_part: NumberField::kpoly_trim(q),
        hermite_terms,
        log_terms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::risch::exp_case::AlgebraicExtension;

    /// K = ℚ(√2), modulus t² − 2.
    fn k_sqrt2() -> (NumberField, AlgebraicExtension) {
        let field = NumberField::new(vec![
            Rational::from(-2),
            Rational::from(0),
            Rational::from(1),
        ]);
        let ext = AlgebraicExtension::SingleSqrt {
            d: 2,
            sqrt_expr: crate::kernel::ExprId(0), // unused by the arithmetic path
        };
        (field, ext)
    }

    /// `√2` as a K-element [0, 1].
    fn sqrt2_elem() -> KElem {
        vec![Rational::from(0), Rational::from(1)]
    }

    #[test]
    fn k_sqrt_perfect_squares() {
        let (field, _ext) = k_sqrt2();
        // 2 = (√2)² ⇒ sqrt(2) = √2 in K.
        let two = field.from_int(2);
        let s = k_sqrt_quadratic(&field, 2, &two).expect("2 is a square in Q(sqrt2)");
        assert!(NumberField::kpoly_eq(
            &[field.mul(&s, &s)],
            &[field.reduce(&two)]
        ));

        // 9 = 3² (rational square).
        let nine = field.from_int(9);
        let s = k_sqrt_quadratic(&field, 2, &nine).expect("9 is a perfect square");
        assert!(NumberField::kpoly_eq(
            &[field.mul(&s, &s)],
            &[field.reduce(&nine)]
        ));

        // (1+√2)² = 3 + 2√2 ⇒ sqrt(3+2√2) = 1+√2.
        let one_plus_sqrt2 = field.add(&field.from_int(1), &sqrt2_elem());
        let target = field.mul(&one_plus_sqrt2, &one_plus_sqrt2);
        let s = k_sqrt_quadratic(&field, 2, &target).expect("(1+sqrt2)^2 is a K-square");
        let s_sq = field.mul(&s, &s);
        assert!(NumberField::kpoly_eq(&[s_sq], &[field.reduce(&target)]));
    }

    #[test]
    fn k_sqrt_non_square() {
        let (field, _ext) = k_sqrt2();
        // 3 is not a square in Q(sqrt2): N(3) = 9, but Q(sqrt2) has no
        // element of norm... actually verify directly: no (p+q√2) with
        // p^2+2q^2=3 and 2pq=0 has rational p,q (p=√3 or q=√(3/2), neither rational).
        let three = field.from_int(3);
        assert!(k_sqrt_quadratic(&field, 2, &three).is_none());
    }

    #[test]
    fn x_times_x_plus_sqrt2_log_terms() {
        // ∫ 1/(x·(x+√2)) dx = (1/√2)·[log(x) − log(x+√2)]
        let (field, ext) = k_sqrt2();
        let sqrt2 = sqrt2_elem();

        // den = x*(x+sqrt2) = x^2 + sqrt2*x  (monic)
        let den: KPoly = vec![NumberField::k_zero(), sqrt2.clone(), field.from_int(1)];
        // num = 1
        let num: KPoly = vec![field.from_int(1)];

        let result = integrate_k_rational_with_logs(&field, &ext, &num, &den)
            .expect("x(x+sqrt2) splits into K-linear factors");

        assert!(NumberField::kdeg(&result.poly_part) < 0); // no polynomial part
        assert!(result.hermite_terms.is_empty()); // squarefree denom — no Hermite part
        assert_eq!(result.log_terms.len(), 2);

        // Roots should be {0, -sqrt2}; residues should be {1/sqrt2, -1/sqrt2}.
        let sqrt2_inv = field.inv(&sqrt2).expect("sqrt2 invertible");
        let neg_sqrt2_inv = field.neg(&sqrt2_inv);

        let mut found_zero = false;
        let mut found_neg_sqrt2 = false;
        for (residue, root) in &result.log_terms {
            if NumberField::kpoly_eq(std::slice::from_ref(root), &[NumberField::k_zero()]) {
                found_zero = true;
                assert!(NumberField::kpoly_eq(
                    std::slice::from_ref(residue),
                    std::slice::from_ref(&sqrt2_inv)
                ));
            } else if NumberField::kpoly_eq(std::slice::from_ref(root), &[field.neg(&sqrt2)]) {
                found_neg_sqrt2 = true;
                assert!(NumberField::kpoly_eq(
                    std::slice::from_ref(residue),
                    std::slice::from_ref(&neg_sqrt2_inv)
                ));
            }
        }
        assert!(found_zero && found_neg_sqrt2);
    }

    #[test]
    fn polynomial_denominator_declines() {
        // den has K-degree 0 → not a genuine rational function.
        let (field, ext) = k_sqrt2();
        let den: KPoly = vec![field.from_int(1)];
        let num: KPoly = vec![field.from_int(1)];
        assert!(integrate_k_rational_with_logs(&field, &ext, &num, &den).is_none());
    }

    #[test]
    fn exact_polynomial_quotient_declines() {
        // num = den ⇒ exact division, no logs needed.
        let (field, ext) = k_sqrt2();
        let den: KPoly = vec![NumberField::k_zero(), sqrt2_elem(), field.from_int(1)];
        let num = den.clone();
        assert!(integrate_k_rational_with_logs(&field, &ext, &num, &den).is_none());
    }

    #[test]
    fn repeated_factor_hermite_reduces() {
        // den = (x+sqrt2)^2 = x^2 + 2sqrt2 x + 2  (not squarefree)
        // ∫ 1/(x+sqrt2)^2 dx = -1/(x+sqrt2)  (no logs needed).
        let (field, ext) = k_sqrt2();
        let sqrt2 = sqrt2_elem();
        let two_sqrt2 = field.mul(&field.from_int(2), &sqrt2);
        let den: KPoly = vec![field.from_int(2), two_sqrt2, field.from_int(1)];
        let num: KPoly = vec![field.from_int(1)];
        let result = integrate_k_rational_with_logs(&field, &ext, &num, &den)
            .expect("Hermite reduction handles the repeated K-factor");

        assert!(NumberField::kdeg(&result.poly_part) < 0); // no polynomial part
        assert!(result.log_terms.is_empty()); // exact Hermite reduction, no logs
        assert_eq!(result.hermite_terms.len(), 1);

        let (b, v_pow, p) = &result.hermite_terms[0];
        assert_eq!(*p, 1);
        // B = -1 (constant), V^1 = x + sqrt2.
        assert!(NumberField::kpoly_eq(b, &[field.neg(&field.from_int(1))]));
        let v: KPoly = vec![sqrt2.clone(), field.from_int(1)];
        assert!(NumberField::kpoly_eq(v_pow, &v));
    }

    #[test]
    fn repeated_factor_with_log_remainder() {
        // den = x*(x+sqrt2)^2.  num = 1.
        // Partial fractions over moduli {x, (x+sqrt2)^2}:
        // 1/(x(x+sqrt2)^2) = A/x + B/(x+sqrt2)^2  with A = 1/(sqrt2)^2 = 1/2,
        // and the (x+sqrt2)^2 part Hermite-reduces to a rational term plus a
        // squarefree x+sqrt2 remainder.
        let (field, ext) = k_sqrt2();
        let sqrt2 = sqrt2_elem();
        let two_sqrt2 = field.mul(&field.from_int(2), &sqrt2);

        // (x+sqrt2)^2 = x^2 + 2sqrt2 x + 2
        let v_sq: KPoly = vec![field.from_int(2), two_sqrt2.clone(), field.from_int(1)];
        // x * (x+sqrt2)^2 = x^3 + 2sqrt2 x^2 + 2x
        let den: KPoly = field.kpoly_mul(&v_sq, &[NumberField::k_zero(), field.from_int(1)]);
        let num: KPoly = vec![field.from_int(1)];

        let result = integrate_k_rational_with_logs(&field, &ext, &num, &den)
            .expect("x*(x+sqrt2)^2 splits with one repeated K-factor");

        assert!(NumberField::kdeg(&result.poly_part) < 0); // no polynomial part
        assert_eq!(result.hermite_terms.len(), 1);
        // The squarefree remainder log_terms should be nonempty (x and x+sqrt2
        // both contribute logs).
        assert_eq!(result.log_terms.len(), 2);
    }

    #[test]
    fn irreducible_quadratic_over_k_declines() {
        // den = x^2 + 1 over Q(sqrt2): discriminant -4, sqrt(-4) not in Q(sqrt2).
        let (field, ext) = k_sqrt2();
        let den: KPoly = vec![field.from_int(1), NumberField::k_zero(), field.from_int(1)];
        let num: KPoly = vec![field.from_int(1)];
        assert!(integrate_k_rational_with_logs(&field, &ext, &num, &den).is_none());
    }
}
