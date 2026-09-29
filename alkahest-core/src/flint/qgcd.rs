//! Monic GCD in `ℚ[x]` through FLINT's modular `fmpz_poly_gcd`.
//!
//! Several subsystems keep their own dense `ℚ[x]` representation (a
//! `Vec<rug::Rational>`, ascending) and used to compute GCDs with the textbook
//! Euclidean algorithm over `ℚ`. That is quadratic in coefficient operations,
//! but the coefficients swell through the remainder sequence and every step is
//! a canonicalising `rug::Rational` operation that pays a bignum GCD. Clearing
//! denominators once and handing the integer problem to FLINT is orders of
//! magnitude faster (Risch: 11.5 s → 0.3 s on a degree-80 input; summation's
//! `RatUniPoly::gcd`: 720 ms → 0.04 ms at degree 80).
//!
//! The answer is identical, not merely equivalent: the monic GCD of two
//! polynomials over a field is unique, and clearing denominators multiplies
//! each input by a nonzero rational, which cannot change it.

use rug::{Integer, Rational};

/// Trailing zeros removed.
fn trimmed(p: &[Rational]) -> &[Rational] {
    let n = p.iter().rposition(|c| *c != 0).map_or(0, |i| i + 1);
    &p[..n]
}

/// `p / lc(p)`; the empty vector for zero.
fn monic(p: &[Rational]) -> Vec<Rational> {
    match p.last() {
        None => Vec::new(),
        Some(lc) if *lc == 1 => p.to_vec(),
        Some(lc) => {
            let inv = lc.clone().recip();
            p.iter().map(|c| Rational::from(c * &inv)).collect()
        }
    }
}

/// The primitive-up-to-content integer associate of a nonzero, trimmed
/// `ℚ`-polynomial: every coefficient multiplied by the lcm of the denominators.
fn to_fmpz(p: &[Rational]) -> super::FlintPoly {
    let mut l = Integer::from(1);
    for c in p.iter().filter(|c| **c != 0) {
        l.lcm_mut(c.denom());
    }
    let ints: Vec<Integer> = p
        .iter()
        .map(|c| c.numer() * Integer::from(&l / c.denom()))
        .collect();
    super::FlintPoly::from_rug_coefficients(&ints)
}

/// Monic `gcd(a, b)` over `ℚ`, ascending coefficients.
///
/// Inputs may carry trailing zeros. `gcd(0, 0)` is the empty vector (the zero
/// polynomial); `gcd(0, p)` is `p` made monic; otherwise the result is monic
/// with no trailing zeros (so `[1]` when the inputs are coprime).
pub(crate) fn qpoly_gcd_monic(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let a = trimmed(a);
    let b = trimmed(b);
    if a.is_empty() {
        return monic(b);
    }
    if b.is_empty() {
        return monic(a);
    }
    if a.len() == 1 || b.len() == 1 {
        return vec![Rational::from(1)];
    }
    let g = to_fmpz(a).gcd(&to_fmpz(b));
    let coeffs: Vec<Rational> = (0..g.length())
        .map(|i| Rational::from(g.get_coeff_flint(i).to_rug()))
        .collect();
    monic(trimmed(&coeffs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Textbook Euclid over ℚ — the reference.
    fn euclid(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
        let mut a = trimmed(a).to_vec();
        let mut b = trimmed(b).to_vec();
        while !b.is_empty() {
            // a mod b
            let lb = b.last().unwrap().clone();
            while a.len() >= b.len() && !a.is_empty() {
                let t = Rational::from(a.last().unwrap() / &lb);
                let s = a.len() - b.len();
                for (j, bj) in b.iter().enumerate() {
                    a[s + j] -= Rational::from(&t * bj);
                }
                let n = trimmed(&a).len();
                a.truncate(n);
            }
            std::mem::swap(&mut a, &mut b);
        }
        monic(&a)
    }

    fn q(v: &[(i64, i64)]) -> Vec<Rational> {
        v.iter().map(|&(n, d)| Rational::from((n, d))).collect()
    }

    fn mul(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut out = vec![Rational::new(); a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                out[i + j] += Rational::from(x * y);
            }
        }
        out
    }

    #[test]
    fn edge_cases() {
        let z: Vec<Rational> = Vec::new();
        let zz = q(&[(0, 1), (0, 1)]);
        assert!(qpoly_gcd_monic(&z, &z).is_empty());
        assert!(qpoly_gcd_monic(&zz, &z).is_empty());
        let p = q(&[(1, 2), (3, 1), (-4, 3)]);
        assert_eq!(qpoly_gcd_monic(&z, &p), monic(&p));
        assert_eq!(qpoly_gcd_monic(&p, &zz), monic(&p));
        assert_eq!(qpoly_gcd_monic(&q(&[(-5, 7)]), &p), q(&[(1, 1)]));
        assert_eq!(qpoly_gcd_monic(&q(&[(-5, 7)]), &z), q(&[(1, 1)]));
        // (x - 1/2)(x + 3) and (x - 1/2)(2x - 7)
        let a = mul(&q(&[(-1, 2), (1, 1)]), &q(&[(3, 1), (1, 1)]));
        let b = mul(&q(&[(-1, 2), (1, 1)]), &q(&[(-7, 1), (2, 1)]));
        assert_eq!(qpoly_gcd_monic(&a, &b), q(&[(-1, 2), (1, 1)]));
    }

    fn arb_qpoly(max_len: usize) -> impl Strategy<Value = Vec<Rational>> {
        prop::collection::vec((-50i64..50, 1i64..12), 0..max_len)
            .prop_map(|v| v.into_iter().map(|(n, d)| Rational::from((n, d))).collect())
    }

    proptest! {
        #[test]
        fn matches_euclid(a in arb_qpoly(7), b in arb_qpoly(7), c in arb_qpoly(5)) {
            // A shared factor `c` makes nontrivial GCDs the common case.
            let ac = mul(&a, &c);
            let bc = mul(&b, &c);
            prop_assert_eq!(qpoly_gcd_monic(&ac, &bc), euclid(&ac, &bc));
            prop_assert_eq!(qpoly_gcd_monic(&a, &b), euclid(&a, &b));
        }
    }

    /// `cargo test --release -p alkahest-cas qgcd_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn qgcd_timing() {
        // Warm FLINT up so the first row does not carry its one-time setup.
        let _ = qpoly_gcd_monic(
            &[Rational::from(1), Rational::from(1)],
            &[Rational::from(2), Rational::from(1)],
        );
        for &d in &[4usize, 8, 20, 40, 80] {
            let gen = |seed: i64, len: usize| -> Vec<Rational> {
                (0..len)
                    .map(|k| {
                        let v = ((k as i64 * 7919 + seed) % 41) - 20;
                        Rational::from((if v == 0 { 1 } else { v }, (k as i64 % 5) + 1))
                    })
                    .collect()
            };
            let c = gen(3, d / 2 + 1);
            let a = mul(&gen(11, d / 2 + 1), &c);
            let b = mul(&gen(17, d / 2 + 1), &c);
            let reps = if d <= 8 { 2000 } else { 5 };
            let t = std::time::Instant::now();
            let mut x = Vec::new();
            for _ in 0..reps {
                x = euclid(&a, &b);
            }
            let old = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let t = std::time::Instant::now();
            let mut y = Vec::new();
            for _ in 0..reps {
                y = qpoly_gcd_monic(&a, &b);
            }
            let new = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            assert_eq!(x, y);
            println!(
                "gcd deg {}: euclid {old:.3} ms, flint {new:.3} ms",
                a.len() - 1
            );
        }
    }
}
