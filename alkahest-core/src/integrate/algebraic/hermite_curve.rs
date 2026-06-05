//! Hermite reduction on an algebraic curve — Risch milestone **M3 / P3**.
//!
//! Given an integrand `f ∈ ℚ(x)(y)` (`F(x,y)=0`), Hermite reduction writes
//! `∫ f dx = g + ∫ h dx` where `g ∈ ℚ(x)(y)` is the **algebraic part** and `h`
//! has only **simple poles** (a squarefree denominator over the curve — a
//! differential of the third kind).  `∫ h dx` is then the logarithmic part (MC).
//!
//! For a **simple radical** `yⁿ = a(x)` the integral basis `wᵢ = yⁱ/dᵢ`
//! diagonalizes the derivation: `wᵢ' = ωᵢ·wᵢ` with
//! `ωᵢ = i·a'/(n·a) − dᵢ'/dᵢ ∈ ℚ(x)`.  Hermite then **decouples** into `n`
//! independent *twisted* scalar Hermite reductions — for the operator
//! `L = d/dx + ωᵢ` — one per basis component (Bronstein, *Symbolic Integration
//! Tutorial* §3.2, eq 12).  The twist's pole at a branch point is handled
//! automatically by the `V·ωᵢ` term, so this is correct including at the
//! branch locus.
//!
//! Sound by construction: the result is accepted only after the exact field
//! identity `g' + h = f` is verified, and each `h` component is checked to have a
//! squarefree denominator.
//!
//! Scope: simple radical extensions (the diagonal case).  The general
//! (non-diagonal) integral basis needs the coupled derivation matrix and is the
//! documented follow-up — its substrate (`integral_basis`, `is_integral`) exists.

use rug::Rational;

use super::super::risch::alg_field::{AlgElem, AlgExtension, RatFn, RationalFunctionField};
use super::super::risch::number_field::{mod_inverse, CoeffField};
use super::super::risch::poly_rde::{degree, poly_deriv, poly_mul, trim, QPoly};
use super::super::risch::rational_rde::{poly_div_exact, poly_gcd};
use super::integral_basis::{radical_integral_basis, squarefree_factors};

/// Hermite reduction of `∫ f dx` on the curve `yⁿ = a(x)`.  Returns `(g, h)` with
/// `f = g' + h` and every component of `h` having a squarefree denominator
/// (simple poles).  `None` if the shape is unsupported or verification fails.
pub fn hermite_reduce_radical(
    n: usize,
    a: &QPoly,
    integrand: &AlgElem,
) -> Option<(AlgElem, AlgElem)> {
    if n < 2 {
        return None;
    }
    let f = RationalFunctionField;
    let basis = radical_integral_basis(n, a)?;
    let ext = AlgExtension::radical(n, a);

    // dᵢ = denominator of the basis element wᵢ = yⁱ/dᵢ.
    let d: Vec<QPoly> = (0..n)
        .map(|i| {
            basis[i]
                .get(i)
                .map(|c| c.denom().clone())
                .unwrap_or_else(|| vec![Rational::from(1)])
        })
        .collect();

    // Integrand in the w-basis: f = Σ fᵢ yⁱ = Σ (fᵢ·dᵢ) wᵢ  (diagonal basis change).
    let a_prime = poly_deriv(a);
    let mut g_w = vec![RatFn::int(0); n];
    let mut h_w = vec![RatFn::int(0); n];
    for i in 0..n {
        let fi = integrand.get(i).cloned().unwrap_or_else(|| RatFn::int(0));
        if fi.numer().is_empty() {
            continue;
        }
        let coord = f.mul(&fi, &RatFn::from_poly(&d[i])); // fᵢ·dᵢ
                                                          // ωᵢ = i·a'/(n·a) − dᵢ'/dᵢ.
        let omega = omega_i(i, n, a, &a_prime, &d[i]);
        let (gi, hi) = twisted_hermite(&coord, &omega)?;
        g_w[i] = gi;
        h_w[i] = hi;
    }

    // Back to the power basis: g = Σ gᵢ wᵢ = Σ (gᵢ/dᵢ) yⁱ.
    let to_power = |w: &[RatFn]| -> AlgElem {
        w.iter()
            .enumerate()
            .map(|(i, gi)| f.mul(gi, &RatFn::new(vec![Rational::from(1)], d[i].clone())))
            .collect()
    };
    let g = to_power(&g_w);
    let h = to_power(&h_w);

    // Soundness gate 1: every h component has a squarefree denominator.
    for hi in &h {
        let den = hi.denom();
        if degree(&poly_gcd(den, &poly_deriv(den))) > 0 {
            return None;
        }
    }
    // Soundness gate 2: g' + h == f exactly in the field.
    let lhs = ext.add(&ext.derivation(&g), &h);
    if !ext.elem_eq(&lhs, integrand) {
        return None;
    }
    Some((g, h))
}

/// `ωᵢ = i·a'/(n·a) − dᵢ'/dᵢ`, the basis-derivative coefficient `wᵢ' = ωᵢ wᵢ`.
fn omega_i(i: usize, n: usize, a: &QPoly, a_prime: &QPoly, di: &QPoly) -> RatFn {
    let f = RationalFunctionField;
    let scale = RatFn::new(
        vec![Rational::from(i as i64)],
        vec![Rational::from(n as i64)],
    );
    let log_a = RatFn::new(a_prime.clone(), a.clone()); // a'/a
    let term1 = f.mul(&scale, &log_a);
    if degree(di) < 1 {
        return term1; // dᵢ = 1 ⇒ dᵢ'/dᵢ = 0
    }
    let log_d = RatFn::new(poly_deriv(di), di.clone()); // dᵢ'/dᵢ
    f.add(&term1, &f.neg(&log_d))
}

/// Twisted scalar Hermite reduction for `L = d/dx + ω`: returns `(g, h)` with
/// `c = L(g) + h`, `h` having a squarefree denominator.
fn twisted_hermite(c: &RatFn, omega: &RatFn) -> Option<(RatFn, RatFn)> {
    let f = RationalFunctionField;
    let mut cur = c.clone();
    let mut g = RatFn::int(0);
    // Each step lowers one repeated factor's multiplicity by one.
    let cap = 4 * (degree(c.denom()).max(0) as usize) + 8;
    for _ in 0..cap {
        let den = cur.denom().clone();
        // Find the highest-multiplicity squarefree factor V (mult M ≥ 2).
        let sqf = squarefree_factors(&den);
        let Some((v, m)) = sqf
            .iter()
            .enumerate()
            .rev()
            .find(|(k, p)| *k + 1 >= 2 && degree(p) >= 1)
            .map(|(k, p)| (p.clone(), k + 1))
        else {
            break; // denominator squarefree → done
        };

        // B ≡ (A/U)·inv(ωV − (M−1)V') (mod V), with U = den / V^M, A = numer(cur).
        let vm = poly_pow(&v, m as u32);
        let u = poly_div_exact(&den, &vm);
        let num = cur.numer().clone();
        let au = poly_mod(&poly_mul(&num, &mod_inverse(&u, &v)?), &v); // A/U mod V

        // K = ωV − (M−1)V'  (a ℚ(x) element, regular at V), reduced mod V.
        let v_rf = RatFn::from_poly(&v);
        let vp_rf = RatFn::from_poly(&poly_scale(
            &poly_deriv(&v),
            &Rational::from((m - 1) as i64),
        ));
        let k_rf = f.add(&f.mul(omega, &v_rf), &f.neg(&vp_rf));
        let k_mod = reduce_mod_v(&k_rf, &v)?;
        let k_inv = mod_inverse(&k_mod, &v)?;

        let b = poly_mod(&poly_mul(&au, &k_inv), &v);
        if trim(b.clone()).is_empty() {
            break; // no reduction possible at this place
        }

        // g += B / V^{M−1};  cur -= L(B/V^{M−1}).
        let term = RatFn::new(b.clone(), poly_pow(&v, (m - 1) as u32));
        g = f.add(&g, &term);
        let l_term = f.add(&f.derivation(&term), &f.mul(omega, &term));
        let next = f.add(&cur, &f.neg(&l_term));
        // Guard against non-progress.
        if degree(next.denom()) >= degree(cur.denom()) && next != RatFn::int(0) {
            // The V-power should strictly drop; if not, stop to stay sound.
            cur = next;
            break;
        }
        cur = next;
    }
    Some((g, cur))
}

/// Reduce a `ℚ(x)` element `r = num/den` modulo `V` (requires `gcd(den, V) = 1`):
/// `num · den⁻¹ mod V`.
fn reduce_mod_v(r: &RatFn, v: &QPoly) -> Option<QPoly> {
    let inv = mod_inverse(r.denom(), v)?;
    Some(poly_mod(&poly_mul(r.numer(), &inv), v))
}

fn poly_mod(a: &QPoly, m: &QPoly) -> QPoly {
    // Remainder of a ÷ m over ℚ[x].
    let (_, rem) = poly_divrem(a, m);
    rem
}

/// `a = q·b + r`, `deg r < deg b`, over `ℚ[x]`.
fn poly_divrem(a: &QPoly, b: &QPoly) -> (QPoly, QPoly) {
    let b = trim(b.clone());
    let bd = degree(&b);
    let mut r = trim(a.clone());
    if bd < 0 {
        return (Vec::new(), r);
    }
    let lc = b[bd as usize].clone();
    let mut q = vec![Rational::from(0); (degree(&r) - bd + 1).max(0) as usize];
    while degree(&r) >= bd && !r.is_empty() {
        let rd = degree(&r);
        let shift = (rd - bd) as usize;
        let factor = r[rd as usize].clone() / &lc;
        if (shift as i64) < q.len() as i64 {
            q[shift] = factor.clone();
        }
        for (i, bc) in b.iter().enumerate() {
            r[shift + i] -= factor.clone() * bc;
        }
        r = trim(r);
    }
    (trim(q), r)
}

fn poly_pow(p: &QPoly, e: u32) -> QPoly {
    let mut acc = vec![Rational::from(1)];
    for _ in 0..e {
        acc = poly_mul(&acc, p);
    }
    acc
}

fn poly_scale(p: &QPoly, s: &Rational) -> QPoly {
    p.iter().map(|c| c.clone() * s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }
    fn rf(num: &[i64], den: &[i64]) -> RatFn {
        RatFn::new(qp(num), qp(den))
    }

    /// ∫ y/x³ dx on y² = x : fully algebraic, g = −⅔ y/x², h = 0.
    #[test]
    fn sqrt_double_pole_fully_reduces() {
        // integrand y/x³ = AlgElem [0, 1/x³].
        let integrand = vec![RatFn::int(0), rf(&[1], &[0, 0, 0, 1])];
        let (g, h) = hermite_reduce_radical(2, &qp(&[0, 1]), &integrand).expect("reduce");
        // h = 0.
        assert!(h.iter().all(|c| c.numer().is_empty()));
        // g = −⅔ y/x²  ⇒  component 1 = −2/3 / x².
        assert_eq!(g[1], RatFn::new(qp(&[-2]), qp(&[0, 0, 3])));
    }

    /// ∫ y/((x−1)·x) dx on y²=x : already simple poles ⇒ g = 0, h = f.
    #[test]
    fn sqrt_simple_pole_untouched() {
        // y/((x-1)x) = AlgElem [0, 1/((x-1)x)] ; (x-1)x = x²−x.
        let integrand = vec![RatFn::int(0), rf(&[1], &[0, -1, 1])];
        let (g, h) = hermite_reduce_radical(2, &qp(&[0, 1]), &integrand).expect("reduce");
        assert!(g.iter().all(|c| c.numer().is_empty())); // g = 0
        assert_eq!(h[1], rf(&[1], &[0, -1, 1])); // h = f
    }

    /// Mixed: ∫ y/(x²(x−1)) dx on y²=x reduces the x² pole, leaving simple poles.
    /// Verified by the exact `g' + h = f` gate inside the reducer.
    #[test]
    fn sqrt_mixed_reduction() {
        // x²(x-1) = x³ − x².
        let integrand = vec![RatFn::int(0), rf(&[1], &[0, 0, -1, 1])];
        let (g, h) = hermite_reduce_radical(2, &qp(&[0, 1]), &integrand).expect("reduce");
        // h has squarefree denominator (gate) and the algebraic part is nontrivial.
        assert!(!g.iter().all(|c| c.numer().is_empty()));
        for hi in &h {
            let den = hi.denom();
            assert!(degree(&poly_gcd(den, &poly_deriv(den))) <= 0);
        }
    }

    /// Degree-3 radical: ∫ y²/x⁴ dx on y³ = x reduces the x⁴ pole.
    #[test]
    fn cbrt_reduction() {
        // y²/x⁴ = AlgElem [0, 0, 1/x⁴].
        let integrand = vec![RatFn::int(0), RatFn::int(0), rf(&[1], &[0, 0, 0, 0, 1])];
        let (g, h) = hermite_reduce_radical(3, &qp(&[0, 1]), &integrand).expect("reduce");
        // y²/x⁴ = x^{2/3}/x⁴ = x^{2/3-4} ; ∫ = x^{2/3-3}/(2/3-3) = (-3/7) x^{-7/3}.
        // x^{-7/3} = y²/x³ ⇒ g component 2 = (-3/7)/x³, h = 0.
        assert!(h.iter().all(|c| c.numer().is_empty()));
        assert_eq!(g[2], RatFn::new(qp(&[-3]), qp(&[0, 0, 0, 7])));
    }
}
