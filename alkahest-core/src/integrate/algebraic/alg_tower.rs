//! Primitive element of a quadratic tower `ℚ(α)[w]/(w² − c)` — substrate for the
//! **algebraic-base-point** logarithmic part (Risch **MC**).
//!
//! When the integrand `(B·y) dx` (`y² = a(x)`) has a pole at an **algebraic**
//! base point `α` (a root of an irreducible `q` of degree `d ≥ 2`), the residue
//! `r0 ± r1·√a(α)` lives in the tower `ℚ(α)(√a(α)) = ℚ(α)[w]/(w² − c)`,
//! `c = a(α) ∈ ℚ(α)`, of degree `2d` over `ℚ`.  Trager's ℚ-basis criterion then
//! needs the residues and the place coordinates expressed in **one** number
//! field; this module builds a primitive element `θ` of that tower and re-expresses
//! `α` and `w` (hence everything) as elements of `ℚ[θ]/M(θ)`.
//!
//! [`primitive_element`] returns `(M, α(θ), w(θ))` with `M` the minimal
//! polynomial of `θ = w + λα` (degree `2d`), and `α`, `w` as `QPoly`s of degree
//! `< 2d`.  Pure exact rational linear algebra in the tower (`ℚ(α)²` arithmetic).
//!
//! **Status:** this is the field-construction substrate only.  Wiring it into a
//! `NonElementary` / `Principal` *verdict* additionally requires reducing the
//! residue-component divisors at primes with a **Galois-consistent labeling** of
//! the `2d` conjugate places (the conjugate-divisor reduction over a generally
//! non-Galois field) — the genuine remaining step, not provided here.

use rug::Rational;

use super::super::risch::poly_rde::{degree, poly_add, poly_mul, poly_scale, trim, QPoly};
use super::super::risch::rational_rde::poly_divrem;

/// A tower element `p0(α) + p1(α)·w`, with `p0, p1 ∈ ℚ[α]/q` (`QPoly` of degree
/// `< d`).
type Tow = (QPoly, QPoly);

/// `a mod q` over `ℚ[x]`.
fn qmod(a: &QPoly, q: &QPoly) -> QPoly {
    trim(poly_divrem(a, q).1)
}

/// Tower multiplication: `(u0+u1 w)(v0+v1 w) = (u0v0 + c·u1v1) + (u0v1+u1v0) w`,
/// reduced mod `q` (and `w² = c`).
fn tmul(u: &Tow, v: &Tow, q: &QPoly, c: &QPoly) -> Tow {
    let p0 = qmod(
        &poly_add(&poly_mul(&u.0, &v.0), &poly_mul(c, &poly_mul(&u.1, &v.1))),
        q,
    );
    let p1 = qmod(&poly_add(&poly_mul(&u.0, &v.1), &poly_mul(&u.1, &v.0)), q);
    (p0, p1)
}

/// Flatten a tower element to a `ℚ`-vector of length `2d` in the basis
/// `{1, α, …, α^{d−1}, w, αw, …, α^{d−1}w}`.
fn tflat(u: &Tow, d: usize) -> Vec<Rational> {
    let mut v = vec![Rational::from(0); 2 * d];
    for i in 0..d {
        if let Some(c) = u.0.get(i) {
            v[i] = c.clone();
        }
        if let Some(c) = u.1.get(i) {
            v[d + i] = c.clone();
        }
    }
    v
}

/// Build a primitive element `θ = w + λα` of `ℚ(α)[w]/(w² − c)` (`q` = minimal
/// polynomial of `α`, monic, `deg q = d ≥ 2`; `c = a(α) ∈ ℚ[α]/q`).  Returns
/// `(M, α_in_θ, w_in_θ)`: `M` = minimal polynomial of `θ` (monic, degree `2d`),
/// and `α`, `w` as elements of `ℚ[t]/M` (`QPoly`s of degree `< 2d`).  `None` if
/// no small `λ` yields a primitive element (should not happen for a genuine
/// quadratic tower).
pub(crate) fn primitive_element(q: &QPoly, c: &QPoly) -> Option<(QPoly, QPoly, QPoly)> {
    let d = degree(q);
    if d < 2 {
        return None;
    }
    let d = d as usize;
    let n = 2 * d;
    let one: Tow = (vec![Rational::from(1)], vec![]);
    let alpha: Tow = (vec![Rational::from(0), Rational::from(1)], vec![]); // α = x
    let w: Tow = (vec![], vec![Rational::from(1)]);

    for lambda in 0..=8i64 {
        // θ = w + λα  ⇒  p0 = λα, p1 = 1.
        let theta: Tow = (
            qmod(&poly_scale(&alpha.0, &Rational::from(lambda)), q),
            vec![Rational::from(1)],
        );
        // θ^0 … θ^n.
        let mut powers: Vec<Tow> = vec![one.clone()];
        let mut cur = one.clone();
        for _ in 0..n {
            cur = tmul(&cur, &theta, q, c);
            powers.push(cur.clone());
        }
        // Columns = θ^0 … θ^{n−1} as ℚ^n vectors; solve θ^n = Σ mₖ θ^k.
        let cols: Vec<Vec<Rational>> = (0..n).map(|k| tflat(&powers[k], d)).collect();
        let Some(m) = solve_columns(&cols, &tflat(&powers[n], d), n) else {
            continue; // singular ⇒ θ not primitive at this λ
        };
        // M(t) = t^n − Σ mₖ t^k.
        let mut mm = vec![Rational::from(0); n + 1];
        mm[n] = Rational::from(1);
        for (k, mk) in m.iter().enumerate() {
            mm[k] = -mk.clone();
        }
        let a_in = solve_columns(&cols, &tflat(&alpha, d), n)?;
        let w_in = solve_columns(&cols, &tflat(&w, d), n)?;
        return Some((trim(mm), trim(a_in), trim(w_in)));
    }
    None
}

/// Solve `A·x = rhs` over `ℚ`, where `A`'s columns are `cols` (each length `n`).
/// `None` if `A` is singular.
fn solve_columns(cols: &[Vec<Rational>], rhs: &[Rational], n: usize) -> Option<Vec<Rational>> {
    // Build augmented matrix rows.
    let mut a: Vec<Vec<Rational>> = (0..n)
        .map(|i| {
            let mut row: Vec<Rational> = (0..n).map(|j| cols[j][i].clone()).collect();
            row.push(rhs[i].clone());
            row
        })
        .collect();
    for col in 0..n {
        let piv = (col..n).find(|&r| a[r][col] != 0)?;
        a.swap(col, piv);
        let inv = Rational::from(1) / a[col][col].clone();
        for v in a[col].iter_mut() {
            *v *= &inv;
        }
        for r in 0..n {
            if r != col && a[r][col] != 0 {
                let f = a[r][col].clone();
                // Two distinct rows (`a[r]` updated from `a[col]`): a range loop is
                // the clearest way to index both without splitting the borrow.
                #[allow(clippy::needless_range_loop)]
                for k in col..=n {
                    let s = f.clone() * &a[col][k];
                    a[r][k] -= s;
                }
            }
        }
    }
    Some((0..n).map(|i| a[i][n].clone()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }

    /// `θ` solves its minimal polynomial, and the returned `α(θ), w(θ)` satisfy
    /// the defining relations `α² ≡ (2 ⇐ q=x²−2)` and `w² ≡ c` in `ℚ[t]/M`.
    fn check(q: &QPoly, c: &QPoly) {
        let (m, a_in, w_in) = primitive_element(q, c).expect("primitive element");
        let d = degree(q) as usize;
        assert_eq!(degree(&m) as usize, 2 * d, "deg M = 2d");
        // α satisfies q: q(α(θ)) ≡ 0 mod M.
        let mut qa = QPoly::new();
        let mut pw = vec![Rational::from(1)];
        for qi in q {
            if *qi != 0 {
                qa = poly_add(&qa, &poly_scale(&pw, qi));
            }
            pw = qmod(&poly_mul(&pw, &a_in), &m);
        }
        assert!(degree(&qmod(&qa, &m)) < 0, "q(α(θ)) ≡ 0");
        // w² ≡ c(α(θ)): evaluate c at α(θ), compare to w(θ)².
        let mut ca = QPoly::new();
        let mut pw2 = vec![Rational::from(1)];
        for ci in c {
            if *ci != 0 {
                ca = poly_add(&ca, &poly_scale(&pw2, ci));
            }
            pw2 = qmod(&poly_mul(&pw2, &a_in), &m);
        }
        let w2 = qmod(&poly_mul(&w_in, &w_in), &m);
        assert_eq!(trim(qmod(&ca, &m)), trim(w2), "w² ≡ c(α)");
    }

    /// Tower `ℚ(√2)[w]/(w² − √2)` = `ℚ(2^{1/4})`, degree 4.
    #[test]
    fn quartic_pure_radical() {
        check(&qp(&[-2, 0, 1]), &qp(&[0, 1])); // q = x²−2, c = √2
    }

    /// `ℚ(√2)[w]/(w² − (1+√2))` — `1+√2` is not a square in `ℚ(√2)`, degree 4.
    #[test]
    fn quartic_nontrivial_sheet() {
        check(&qp(&[-2, 0, 1]), &qp(&[1, 1])); // q = x²−2, c = 1+√2
    }

    /// Cubic base: `ℚ(α)[w]/(w² − α)` with `α³ = 2` (`q = x³−2`), degree 6.
    #[test]
    fn sextic_cubic_base() {
        check(&qp(&[-2, 0, 0, 1]), &qp(&[0, 1])); // q = x³−2, c = α
    }
}
