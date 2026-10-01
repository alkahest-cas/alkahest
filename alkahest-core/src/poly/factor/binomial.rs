//! Closed-form factorisation of binomials `a·x^(k+n) + b·x^k` over ℤ.
//!
//! `X^n − Y^n = ∏_{d | n} Φ_d(X, Y)` and `X^n + Y^n = ∏_{d | 2n, d ∤ n} Φ_d(X, Y)`,
//! where `Φ_d(X, Y) = Y^φ(d)·Φ_d(X/Y)` is the homogenised cyclotomic
//! polynomial. Every `Φ_d` is irreducible over ℚ, so with `X = u·x`, `Y = v`
//! and `gcd(u, v) = 1` each `Φ_d(u·x, v)` is an irreducible, primitive
//! polynomial with a positive leading coefficient `u^φ(d)` — exactly the
//! normalisation FLINT's `fmpz_poly_factor` returns.
//!
//! FLINT's own factoriser reaches the same factors through Zassenhaus /
//! van Hoeij recombination, and on `x^n − 1` with many divisors it spends
//! nearly all of its time in LLL (`x^120 − 1`: 40 ms; `x^720 − 1`: 5 s).
//! Building the cyclotomic polynomials directly takes microseconds.
//!
//! The fast path applies only when the primitive part is `u^n·x^n ± v^n`
//! with `u, v ≥ 1` exact `n`-th powers; anything else (and any input whose
//! factors would exceed the size guards) returns `None` and is factored by
//! FLINT as before.

use crate::flint::{FlintInteger, FlintPoly};
use rug::Integer;

/// Factors of degree above this are not built here: the same cap as
/// [`crate::numfield::MAX_CYCLOTOMIC_POLYNOMIAL_DEGREE`] (#414), past which
/// `fmpz_poly_cyclotomic` is refused rather than asked for an allocation it
/// could abort on. Such inputs fall through to the FLINT path, as before.
const MAX_FACTOR_DEGREE: u64 = crate::numfield::MAX_CYCLOTOMIC_POLYNOMIAL_DEGREE;

/// `(prime, exponent)` pairs of `m ≥ 1` by trial division.
fn factor_small(mut m: u64) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    let mut p = 2u64;
    while p.saturating_mul(p) <= m {
        if m % p == 0 {
            let mut e = 0;
            while m % p == 0 {
                m /= p;
                e += 1;
            }
            out.push((p, e));
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if m > 1 {
        out.push((m, 1));
    }
    out
}

/// Every divisor `d` of `∏ pᵢ^eᵢ`, paired with `φ(d)`.
fn divisors_with_phi(primes: &[(u64, u32)]) -> Vec<(u64, u64)> {
    let mut out = vec![(1u64, 1u64)];
    for &(p, e) in primes {
        let len = out.len();
        let mut pk = 1u64;
        for _ in 0..e {
            pk *= p;
            let phi_pk = pk - pk / p;
            for i in 0..len {
                let (d, phi) = out[i];
                out.push((d * pk, phi * phi_pk));
            }
        }
    }
    out
}

/// The exact `n`-th root of `m ≥ 1`, if `m` is a perfect `n`-th power.
fn exact_root(m: &Integer, n: u64) -> Option<Integer> {
    if *m == 1 {
        return Some(Integer::from(1));
    }
    // A perfect n-th power other than 1 has at least n bits.
    if u64::from(m.significant_bits()) < n {
        return None;
    }
    let n32 = u32::try_from(n).ok()?;
    let (root, rem) = m.clone().root_rem(Integer::new(), n32);
    (rem == 0).then_some(root)
}

/// `Φ_d(u·x, v)`, ascending, i.e. `c_i·u^i·v^(φ(d)−i)` for `Φ_d = Σ c_i x^i`.
fn scaled_cyclotomic(d: u64, phi: u64, u: &Integer, v: &Integer) -> FlintPoly {
    let base = FlintPoly::cyclotomic(d);
    if *u == 1 && *v == 1 {
        return base;
    }
    let phi = phi as usize;
    let mut upow = Vec::with_capacity(phi + 1);
    let mut vpow = Vec::with_capacity(phi + 1);
    upow.push(Integer::from(1));
    vpow.push(Integer::from(1));
    for i in 1..=phi {
        upow.push(Integer::from(&upow[i - 1] * u));
        vpow.push(Integer::from(&vpow[i - 1] * v));
    }
    let coeffs: Vec<Integer> = (0..=phi)
        .map(|i| base.get_coeff_flint(i).to_rug() * &upow[i] * &vpow[phi - i])
        .collect();
    FlintPoly::from_rug_coefficients(&coeffs)
}

/// Factor `f = a·x^(k+n) + b·x^k` (`a, b ≠ 0`, `n ≥ 1`) through cyclotomic
/// polynomials, in FLINT's normalisation: `unit = sign(a)·gcd(a, b)` and
/// every factor primitive with a positive leading coefficient.
///
/// `None` when `f` is not such a binomial, its primitive part is not
/// `u^n·x^n ± v^n`, or the factors would pass the size guards; the caller
/// then factors with FLINT. The factor order is unspecified here — the
/// caller sorts into the canonical order.
pub(crate) fn factor_binomial_z(f: &FlintPoly) -> Option<(FlintInteger, Vec<(FlintPoly, u32)>)> {
    let (k, n) = f.binomial_shape()?;
    let n = n as u64;
    let a = f.get_coeff_flint(k + n as usize).to_rug();
    let b = f.get_coeff_flint(k).to_rug();
    let g = Integer::from(a.gcd_ref(&b));
    let unit = if a < 0 { -g.clone() } else { g.clone() };
    // Primitive part `a'·x^n + b'` with `a' > 0`.
    let a1 = Integer::from(&a / &unit);
    let b1 = Integer::from(&b / &unit);
    let u = exact_root(&a1, n)?;
    let v = exact_root(&Integer::from(b1.abs_ref()), n)?;
    let plus = b1 > 0;

    // `x^n + 1` takes the divisors of 2n that do not divide n: with
    // n = 2^e·m (m odd), those are 2^(e+1)·d for d | m.
    let mut primes = factor_small(n);
    if plus {
        match primes.first_mut() {
            Some((2, e)) => *e += 1,
            _ => primes.insert(0, (2, 1)),
        }
    }
    let mut divisors = divisors_with_phi(&primes);
    if plus {
        let two_part = 1u64 << primes[0].1;
        divisors.retain(|&(d, _)| d % two_part == 0);
    }
    if divisors.iter().any(|&(_, phi)| phi > MAX_FACTOR_DEGREE) {
        return None;
    }
    // `Φ_d(u·x, v)` has coefficients up to about `φ(d)·log₂ max(u, v)` bits on
    // top of Φ_d's own height, so the factors together can be far larger than
    // the two-coefficient input. Refuse before building them (FLINT would
    // produce the same factors, just slowly, so the fall-through is exact).
    if u != 1 || v != 1 {
        let bits = u64::from(u.significant_bits().max(v.significant_bits()));
        let total: u64 = divisors.iter().fold(0u64, |acc, &(_, phi)| {
            acc.saturating_add(phi.saturating_mul(phi).saturating_mul(bits))
        });
        if crate::budget::preflight_bytes(total / 8).is_err() {
            return None;
        }
    }

    let mut factors = Vec::with_capacity(divisors.len() + 1);
    if k > 0 {
        let x = FlintPoly::from_coefficients(&[0, 1]);
        factors.push((x, u32::try_from(k).ok()?));
    }
    for &(d, phi) in &divisors {
        factors.push((scaled_cyclotomic(d, phi, &u, &v), 1));
    }
    Some((FlintInteger::from_rug(&unit), factors))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divisors_and_phi() {
        let mut d = divisors_with_phi(&factor_small(12));
        d.sort_unstable();
        assert_eq!(d, vec![(1, 1), (2, 1), (3, 2), (4, 2), (6, 2), (12, 4)]);
        assert_eq!(factor_small(1), vec![]);
        assert_eq!(factor_small(97), vec![(97, 1)]);
    }

    type Factored = (FlintInteger, Vec<(FlintPoly, u32)>);

    /// The pre-fast-path result: FLINT's factoriser, sorted canonically.
    fn reference(f: &FlintPoly) -> Factored {
        let (unit, mut factors) = f.factor_over_z_flint_order().unwrap();
        factors.sort_by(crate::poly::factor::cmp_univariate_factors);
        (unit, factors)
    }

    fn assert_same(f: &FlintPoly, what: &str) {
        assert!(
            factor_binomial_z(f).is_some(),
            "{what}: the fast path should apply"
        );
        let (u_fast, fast) = f.factor_over_z().unwrap();
        let (u_ref, slow) = reference(f);
        assert_eq!(u_fast, u_ref, "{what}: unit");
        assert_eq!(fast.len(), slow.len(), "{what}: number of factors");
        for ((pf, ef), (ps, es)) in fast.iter().zip(&slow) {
            assert!(pf == ps && ef == es, "{what}: {pf:?}^{ef} vs {ps:?}^{es}");
        }
    }

    fn binomial(a: i64, k: usize, n: usize, b: i64) -> FlintPoly {
        let mut c = vec![0i64; k + n + 1];
        c[k + n] = a;
        c[k] = b;
        FlintPoly::from_coefficients(&c)
    }

    /// `x^n ± 1` against FLINT for every `n` up to 300 (primes, prime powers
    /// and the highly composite 120, 180, 240, 300 among them). FLINT needs a
    /// few seconds for the composite ones in this range.
    #[test]
    fn x_n_pm_1_matches_flint() {
        for n in 1..=300usize {
            assert_same(&binomial(1, 0, n, -1), &format!("x^{n} - 1"));
            assert_same(&binomial(1, 0, n, 1), &format!("x^{n} + 1"));
        }
    }

    /// `12x² + 3 = 3·((2x)² + 1)`: a clean binomial once the content is out.
    #[test]
    fn content_is_removed_before_the_power_test() {
        assert_same(&binomial(12, 0, 2, 3), "12x^2 + 3");
        assert_same(&binomial(-64, 2, 5, 486), "-64x^7 + 486x^2");
    }

    /// Content, sign, the `x^k` factor and scaled `u^n·x^n ± v^n` binomials.
    #[test]
    fn scaled_and_shifted_binomials_match_flint() {
        for n in 1..=12usize {
            for k in [0usize, 1, 3] {
                for (u, v) in [(1i64, 1i64), (2, 1), (1, 2), (2, 3), (3, 2)] {
                    let (un, vn) = (u.pow(n as u32), v.pow(n as u32));
                    if un > 1 << 20 || vn > 1 << 20 {
                        continue;
                    }
                    for content in [1i64, -1, 6, -10] {
                        for sign in [1i64, -1] {
                            let f = binomial(content * un, k, n, content * sign * vn);
                            assert_same(
                                &f,
                                &format!("{content}*({un}x^{n} + {}) x^{k}", sign * vn),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Binomials that are not `u^n·x^n ± v^n` are left to FLINT.
    #[test]
    fn unclean_binomials_fall_through() {
        for (a, n, b) in [
            (2i64, 2usize, -1i64),
            (1, 2, -2),
            (4, 4, -1),
            (1, 6, 8),
            (12, 2, 5),
        ] {
            let f = binomial(a, 0, n, b);
            assert!(factor_binomial_z(&f).is_none(), "{a}x^{n} + {b}");
            let (u, fac) = f.factor_over_z().unwrap();
            let (u_ref, fac_ref) = reference(&f);
            assert!(u == u_ref && fac == fac_ref, "{a}x^{n} + {b}");
        }
        // Not binomials at all.
        for c in [&[1i64][..], &[0, 0, 3], &[1, 1, 1], &[0, 1, 0, 1, 1]] {
            assert!(
                factor_binomial_z(&FlintPoly::from_coefficients(c)).is_none(),
                "{c:?}"
            );
        }
    }

    #[test]
    fn roots() {
        assert_eq!(exact_root(&Integer::from(27), 3), Some(Integer::from(3)));
        assert_eq!(exact_root(&Integer::from(28), 3), None);
        assert_eq!(exact_root(&Integer::from(1), 1000), Some(Integer::from(1)));
        assert_eq!(exact_root(&Integer::from(2), 1000), None);
    }
}
