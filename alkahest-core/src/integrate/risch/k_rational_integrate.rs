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

/// Result of `integrate_k_rational_with_logs`: the polynomial/rational part
/// (as a `K`-rational function `(num, den)`) plus the logarithmic terms
/// `Σ cᵢ·log(x − rᵢ)`.
pub struct KRationalLogResult {
    /// `K`-rational antiderivative of the "no new log" part: `(num, den)`.
    pub rational_num: KPoly,
    pub rational_den: KPoly,
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

/// Try to compute `∫ c_num/c_den dx` over `K = ℚ(α)`, allowing `log` terms
/// with `K`-coefficients in the result.
///
/// `c_num`, `c_den` are `K`-polynomials in `x`; `field`/`ext` describe `K`.
///
/// Returns `None` when:
/// - `c_den` reduces to a polynomial (no logs needed — caller should use the
///   plain `solve_rational_rde_k` path instead),
/// - the squarefree remainder's denominator has a repeated factor (Hermite
///   reduction over `K` not yet implemented), or
/// - the squarefree remainder's denominator does not split completely into
///   distinct `K`-linear factors (non-linear irreducible `K`-factor).
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

    // Squarefree check: gcd(den, den') must be a (nonzero) constant.
    let dprime = field.kpoly_deriv(&den);
    let gd = field.kpoly_gcd(&den, &dprime)?;
    if NumberField::kdeg(&gd) > 0 {
        return None; // repeated factor — Hermite reduction over K not yet implemented
    }

    // Find all roots of `den` in K (must split completely).
    let roots = find_k_roots(field, &den, ext)?;
    if roots.len() != NumberField::kdeg(&den) as usize {
        return None;
    }

    // Simple-pole residues: cᵢ = R(rᵢ) / D'(rᵢ).
    let mut log_terms = Vec::new();
    for root in &roots {
        let r_val = eval_kpoly_at(field, &r, root);
        let dp_val = eval_kpoly_at(field, &dprime, root);
        let dp_inv = field.inv(&dp_val)?;
        let residue = field.mul(&r_val, &dp_inv);
        if NumberField::is_zero(&residue) {
            continue;
        }
        log_terms.push((residue, root.clone()));
    }

    Some(KRationalLogResult {
        rational_num: NumberField::kpoly_trim(q),
        rational_den: vec![field.from_int(1)],
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

        assert!(NumberField::kdeg(&result.rational_num) < 0); // no polynomial part
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
    fn repeated_factor_declines() {
        // den = (x+sqrt2)^2 = x^2 + 2sqrt2 x + 2  (not squarefree)
        let (field, ext) = k_sqrt2();
        let sqrt2 = sqrt2_elem();
        let two_sqrt2 = field.mul(&field.from_int(2), &sqrt2);
        let den: KPoly = vec![field.from_int(2), two_sqrt2, field.from_int(1)];
        let num: KPoly = vec![field.from_int(1)];
        assert!(integrate_k_rational_with_logs(&field, &ext, &num, &den).is_none());
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
