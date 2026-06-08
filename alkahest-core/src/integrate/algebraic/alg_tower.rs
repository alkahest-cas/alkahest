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
//! For a **quadratic base** `q = x²−m` whose tower `K = ℚ(√m)[w]/(w²−c)` is
//! Galois over `ℚ`, [`galois_quartic`] additionally returns the four
//! automorphism images of `θ` (each *verified* `M(πⱼ) ≡ 0`), built from
//! quadratic-field square roots ([`sqrt_in_quad`]) — no number-field
//! factorization.  These give the conjugate residues and places (via
//! [`compose_mod`]) that `genus_zero::try_alg_base_log` decomposes and reduces.
//! A non-Galois tower (the conjugate sheet `√(c̄) ∉ K`) is declined.

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

/// `f(g) mod M` — compose polynomials, reduced mod `M` (Horner).
pub(crate) fn compose_mod(f: &QPoly, g: &QPoly, m: &QPoly) -> QPoly {
    let mut acc: QPoly = Vec::new();
    for fi in f.iter().rev() {
        acc = qmod(&poly_mul(&acc, g), m);
        if *fi != 0 {
            acc = poly_add(&acc, &vec![fi.clone()]);
        }
    }
    qmod(&acc, m)
}

/// Exact rational square root, or `None`.
fn rat_sqrt(r: &Rational) -> Option<Rational> {
    if *r < 0 {
        return None;
    }
    let (n, d) = (r.numer().clone(), r.denom().clone());
    let (ns, ds) = (n.clone().sqrt(), d.clone().sqrt());
    if rug::Integer::from(&ns * &ns) == n && rug::Integer::from(&ds * &ds) == d {
        Some(Rational::from((ns, ds)))
    } else {
        None
    }
}

/// Square root of `u + v√m` in `ℚ(√m)`, as `[a, b]` (`= a + b√m`), or `None` if
/// it is not a square there.
fn sqrt_in_quad(u: &Rational, v: &Rational, m: &Rational) -> Option<[Rational; 2]> {
    let zero = Rational::from(0);
    if *v == 0 {
        if let Some(a) = rat_sqrt(u) {
            return Some([a, zero]);
        }
        if let Some(b) = rat_sqrt(&(u.clone() / m)) {
            return Some([zero, b]);
        }
        return None;
    }
    let disc = u.clone() * u - v.clone() * v * m; // u² − v²m, must be a square
    let s = rat_sqrt(&disc)?;
    for sg in [Rational::from(1), Rational::from(-1)] {
        let a2 = (u.clone() + sg * &s) / Rational::from(2);
        if let Some(a) = rat_sqrt(&a2) {
            if a != 0 {
                let b = v.clone() / (Rational::from(2) * &a);
                return Some([a, b]);
            }
        }
    }
    None
}

/// For a **quadratic** base `q = x² − m` and sheet radicand `c = a(α) ∈ ℚ(α)`
/// (`α = √m`), build the degree-4 field `K = ℚ(√m)[w]/(w²−c)` and, **when it is
/// Galois over ℚ**, its four automorphism images of `θ` (as `QPoly`s mod `M`).
/// Returns `(M, α(θ), w(θ), [π₀=θ, π₁, π₂, π₃])`.  `None` if `q` is not `x²−m`,
/// `K` is not degree 4, or `K/ℚ` is not Galois (the conjugate sheet `√(c̄)` is
/// not in `K`) — all sound declines.  Every returned `πⱼ` is **verified**
/// (`M(πⱼ) ≡ 0 (mod M)`), so the result is correct by construction.
pub(crate) fn galois_quartic(q: &QPoly, c: &QPoly) -> Option<(QPoly, QPoly, QPoly, Vec<QPoly>)> {
    if degree(q) != 2 || q.get(1).map(|x| *x != 0).unwrap_or(false) {
        return None; // require q = x² − m
    }
    let m = -q[0].clone(); // α² = m
    let (mm, a_in, w_in) = primitive_element(q, c)?;
    if degree(&mm) != 4 {
        return None;
    }
    let u = c.first().cloned().unwrap_or_else(|| Rational::from(0));
    let v = c.get(1).cloned().unwrap_or_else(|| Rational::from(0));
    // c̄ = u − v√m;  c̄/c = (u²+v²m − 2uv√m)/(u²−v²m).
    let nrm = u.clone() * &u - v.clone() * &v * &m; // N(c)
    if nrm == 0 {
        return None;
    }
    let cbar = [u.clone(), -v.clone()];
    let cbar_over_c = [
        (u.clone() * &u + v.clone() * &v * &m) / &nrm,
        Rational::from(-2) * &u * &v / &nrm,
    ];

    // w_α = √(c̄) ∈ K: either √(c̄/c)·w (g ∈ ℚ(α)) or √(c̄) ∈ ℚ(α).
    let w_alpha: QPoly = if let Some(g) = sqrt_in_quad(&cbar_over_c[0], &cbar_over_c[1], &m) {
        // g(α)·w  with g = g0 + g1·α.
        let g_at = poly_add(
            &vec![g[0].clone()],
            &poly_scale(&a_in, &g[1]), // g1·α(θ)
        );
        qmod(&poly_mul(&g_at, &w_in), &mm)
    } else if let Some(h) = sqrt_in_quad(&cbar[0], &cbar[1], &m) {
        // √(c̄) = h0 + h1·α ∈ ℚ(α).
        poly_add(&vec![h[0].clone()], &poly_scale(&a_in, &h[1]))
    } else {
        return None; // conjugate sheet not in K ⇒ not Galois
    };

    // π₀ = θ;  σ_w(θ) = θ − 2w;  σ_α(θ) = w_α + w − θ;  σ_αw(θ) = −w_α + w − θ.
    let theta = vec![Rational::from(0), Rational::from(1)];
    let two_w = poly_scale(&w_in, &Rational::from(2));
    let sigma_w = qmod(&poly_sub_q(&theta, &two_w), &mm);
    let sigma_a = qmod(&poly_sub_q(&poly_add(&w_alpha, &w_in), &theta), &mm);
    let sigma_aw = qmod(&poly_sub_q(&poly_sub_q(&w_in, &w_alpha), &theta), &mm);
    let autos = vec![theta, sigma_w, sigma_a, sigma_aw];

    // Verify each is a genuine root of M (sound) and they are distinct.
    for (i, pi) in autos.iter().enumerate() {
        if degree(&eval_mod(&mm, pi, &mm)) >= 0 {
            return None; // πᵢ not a root of M ⇒ construction invalid (not Galois)
        }
        for pj in autos.iter().take(i) {
            if trim(pi.clone()) == trim(pj.clone()) {
                return None; // repeated ⇒ fewer than 4 automorphisms
            }
        }
    }
    Some((mm, a_in, w_in, autos))
}

/// `f(β) mod M`.
fn eval_mod(f: &QPoly, beta: &QPoly, m: &QPoly) -> QPoly {
    compose_mod(f, beta, m)
}

fn poly_sub_q(a: &QPoly, b: &QPoly) -> QPoly {
    poly_add(a, &poly_scale(b, &Rational::from(-1)))
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

    /// Galois quartic `ℚ(√2, √3)` (`q = x²−2`, `c = 3` rational): four
    /// automorphisms; `M = t⁴−10t²+1` is the minimal polynomial of `√2+√3`.
    #[test]
    fn galois_quartic_multiquadratic() {
        let (m, _a, _w, autos) = galois_quartic(&qp(&[-2, 0, 1]), &qp(&[3])).expect("Galois");
        assert_eq!(m, qp(&[1, 0, -10, 0, 1])); // t⁴ − 10t² + 1
        assert_eq!(autos.len(), 4);
    }

    /// Non-Galois quartic `ℚ(√2, √(1+5√2))` (`q = x²−2`, `c = 1+5√2`): the
    /// conjugate sheet `√(1−5√2) ∉ K`, so `galois_quartic` declines.
    #[test]
    fn non_galois_quartic_declines() {
        assert!(galois_quartic(&qp(&[-2, 0, 1]), &qp(&[1, 5])).is_none());
    }
}
