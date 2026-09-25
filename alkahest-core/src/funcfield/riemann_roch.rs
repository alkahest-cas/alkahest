//! `dim L(D)` and an explicit basis of `L(D)`, for the imaginary hyperelliptic
//! model with rational places.
//!
//! # The algorithm
//!
//! `L(D) = { u ∈ K* : div(u) + D ≥ 0 } ∪ {0}`.  Two facts make this a finite
//! linear-algebra problem on `y² = a(x)` with `a` squarefree:
//!
//! 1. The integral closure of `ℚ[x]` in `K` is exactly `ℚ[x] ⊕ ℚ[x]·y` — the
//!    functions with no finite pole.  (`a` squarefree is what makes the naïve
//!    basis integral; see [`crate::integrate::algebraic::integral_basis`].)
//! 2. On the **odd**-degree model the two candidate pole orders at infinity,
//!    `−2·deg p` and `−deg a − 2·deg q`, have opposite parity, so
//!    `v_∞(p + q·y) = min(−2·deg p, −deg a − 2·deg q)` *exactly*, with no
//!    cancellation to worry about.
//!
//! So pick `e(x) = Π (x − α)^{eα}` clearing the positive finite part of `D`,
//! where `eα = max_{P|α} ⌈D(P)/c_P⌉` and `c_P = v_P(x − α) ∈ {1, 2}`.  Then
//! `u ∈ L(D)` iff `w = u·e` lies in `ℚ[x] ⊕ ℚ[x]·y` with
//!
//! ```text
//!     deg p ≤ deg e + ⌊n∞/2⌋,
//!     deg q ≤ deg e + ⌊(n∞ − deg a)/2⌋,
//!     v_P(w) ≥ c_P·eα − D(P)   at each finite place P where that is positive.
//! ```
//!
//! The degree bounds cut out a finite-dimensional space of `(p, q)`, and each
//! valuation condition is `k` linear equations over ℚ: expand `x` and `y` as
//! power series in a local uniformiser at `P` and require the first `k`
//! coefficients of `w` to vanish.  The uniformiser is `t = x − α` at an
//! unramified place (where `y` is a square root of `a(α + t)`, computed by the
//! standard quadratic recurrence) and `t = y` at a branch point (where `x − α`
//! is recovered by Newton iteration from `a(α + X) = t²`).
//!
//! Every place that carries a condition is rational: the ones dividing `e` lie
//! over the support of `D`'s positive part, which is rational by construction,
//! and the conjugate of a rational place is rational.  So no quadratic
//! extension is ever needed, and the matrix is over ℚ.
//!
//! # Scope
//!
//! Imaginary (odd-degree) model, rational places.  On an even-degree model
//! `v_∞` is *not* given by the minimum above — the two places at infinity split
//! the pole order — and this construction does not apply; it refuses with
//! `E-FFLD-002`.

use rug::Rational;

use super::divisor::{Divisor, Place};
use super::element::FunctionFieldElement;
use super::error::FunctionFieldError;
use super::util::{eval, linear, nullspace, series_mul, series_of_poly, taylor_shift};
use crate::integrate::risch::poly_rde::{degree, poly_deriv, poly_mul, trim, QPoly};

/// A cap on the size of the linear system, so a wild divisor is a refusal
/// rather than an out-of-memory.
const MAX_UNKNOWNS: usize = 4096;

/// The space `L(D)`: its dimension and an explicit basis.
#[derive(Clone, Debug)]
pub struct RiemannRochSpace {
    divisor: Divisor,
    basis: Vec<FunctionFieldElement>,
}

impl RiemannRochSpace {
    /// `dim_ℚ L(D)`.
    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    /// A ℚ-basis of `L(D)`, each element as `(p + q·y)/e`.
    ///
    /// Every returned `u` satisfies `div(u) + D ≥ 0`; the basis is empty
    /// exactly when `L(D) = 0`.
    pub fn basis(&self) -> &[FunctionFieldElement] {
        &self.basis
    }

    /// The divisor this space was computed for.
    pub fn divisor(&self) -> &Divisor {
        &self.divisor
    }
}

/// `L(D)` — dimension and basis.
///
/// See the module documentation for the algorithm and the scope.
pub fn riemann_roch(d: &Divisor) -> Result<RiemannRochSpace, FunctionFieldError> {
    let field = d.field().clone();
    field.require_imaginary("riemann_roch")?;
    let a = field.curve().clone();
    let da = field.curve_degree() as i64;
    let g = field.genus() as i64;

    let deg_d = d.degree();
    let deg_d_i64 = deg_d.to_i64().ok_or_else(|| FunctionFieldError::TooLarge {
        reason: format!("deg D = {deg_d} does not fit in a machine word"),
    })?;

    // --- the clearing polynomial e(x) -------------------------------------
    // For each x-coordinate carrying a positive multiplicity, the exponent is
    // max over the places above it of ⌈D(P)/c_P⌉.
    let mut exps: Vec<(Rational, i64)> = Vec::new();
    for (x, y, c) in d.finite_terms() {
        if c <= 0 {
            continue;
        }
        let c = c.to_i64().ok_or_else(|| FunctionFieldError::TooLarge {
            reason: format!("multiplicity {c} at ({x}, {y}) does not fit in a machine word"),
        })?;
        let cp: i64 = if y == 0 { 2 } else { 1 };
        let need = (c + cp - 1) / cp; // ⌈c/cp⌉, both positive
        match exps.iter_mut().find(|(v, _)| *v == x) {
            Some(slot) => slot.1 = slot.1.max(need),
            None => exps.push((x, need)),
        }
    }
    exps.sort_by(|p, q| p.0.cmp(&q.0));

    let mut e: QPoly = vec![Rational::from(1)];
    let mut deg_e: i64 = 0;
    for (alpha, k) in &exps {
        for _ in 0..*k {
            e = poly_mul(&e, &linear(alpha));
        }
        deg_e += *k;
    }
    let e = trim(e);

    let n_inf = d
        .infinity_coefficient()
        .to_i64()
        .ok_or_else(|| FunctionFieldError::TooLarge {
            reason: "the multiplicity at infinity does not fit in a machine word".into(),
        })?;

    // --- degree bounds on (p, q) ------------------------------------------
    let dp = deg_e + (n_inf).div_euclid(2);
    let dq = deg_e + (n_inf - da).div_euclid(2);
    let np = if dp < 0 { 0usize } else { (dp + 1) as usize };
    let nq = if dq < 0 { 0usize } else { (dq + 1) as usize };
    let ncols = np + nq;
    if ncols > MAX_UNKNOWNS {
        return Err(FunctionFieldError::TooLarge {
            reason: format!("the Riemann–Roch system would have {ncols} unknowns"),
        });
    }
    if ncols == 0 {
        return finish(d.clone(), Vec::new(), deg_d_i64, g);
    }

    // --- the constraint places and their orders ---------------------------
    // Every place above an x-coordinate dividing e, plus every place where D
    // is negative.  Both families are rational.
    let mut constraints: Vec<(Place, i64)> = Vec::new();
    let push = |place: Place, k: i64, out: &mut Vec<(Place, i64)>| {
        if k > 0 && !out.iter().any(|(p, _)| *p == place) {
            out.push((place, k));
        }
    };
    // A multiplicity past a machine word is a refusal, never a default: a
    // silent `0` here would drop a condition and return a space that is too
    // large, which is the exact failure mode this module exists to avoid.
    let mult = |d: &Divisor, p: &Place| -> Result<i64, FunctionFieldError> {
        let c = d.coefficient(p);
        c.to_i64().ok_or_else(|| FunctionFieldError::TooLarge {
            reason: format!("multiplicity {c} at {p} does not fit in a machine word"),
        })
    };
    for (alpha, k) in &exps {
        let a_alpha = eval(&a, alpha);
        if a_alpha == 0 {
            let p = Place::finite(alpha.clone(), Rational::from(0));
            let need = 2 * k - mult(d, &p)?;
            push(p, need, &mut constraints);
        } else {
            let beta = super::util::rational_sqrt(&a_alpha).ok_or_else(|| {
                FunctionFieldError::SelfCheckFailed {
                    detail: format!(
                        "x = {alpha} carries a positive multiplicity but a({alpha}) = {a_alpha} \
                         is not a rational square"
                    ),
                }
            })?;
            for s in [beta.clone(), -beta] {
                let p = Place::finite(alpha.clone(), s);
                let need = k - mult(d, &p)?;
                push(p, need, &mut constraints);
            }
        }
    }
    for (x, y, c) in d.finite_terms() {
        if c >= 0 {
            continue;
        }
        let p = Place::finite(x.clone(), y.clone());
        let cp: i64 = if y == 0 { 2 } else { 1 };
        let e_alpha = exps
            .iter()
            .find(|(v, _)| *v == x)
            .map(|(_, k)| *k)
            .unwrap_or(0);
        let need = cp * e_alpha - mult(d, &p)?;
        push(p, need, &mut constraints);
    }

    let total_rows: i64 = constraints.iter().map(|(_, k)| *k).sum();
    if total_rows > MAX_UNKNOWNS as i64 {
        return Err(FunctionFieldError::TooLarge {
            reason: format!("the Riemann–Roch system would have {total_rows} equations"),
        });
    }

    // --- assemble the matrix ----------------------------------------------
    let top = np.max(nq);
    let mut rows: Vec<Vec<Rational>> = Vec::with_capacity(total_rows as usize);
    for (place, k) in &constraints {
        let k = *k as usize;
        let (xs, ys) = local_expansions(&a, place, k)?;
        // Self-check: y(t)² must equal a(x(t)) as truncated series.
        let need = top.max(da as usize + 1);
        let xpow = power_series(&xs, need, k);
        let mut a_of_x = vec![Rational::from(0); k];
        for (i, ai) in a.iter().enumerate() {
            if *ai == 0 {
                continue;
            }
            for (n, slot) in a_of_x.iter_mut().enumerate() {
                *slot += Rational::from(ai * &xpow[i][n]);
            }
        }
        let y2 = series_mul(&ys, &ys, k);
        if a_of_x != y2 {
            return Err(FunctionFieldError::SelfCheckFailed {
                detail: format!(
                    "the local expansion at {place} does not satisfy y² = a(x) to {k} terms"
                ),
            });
        }
        let qser: Vec<Vec<Rational>> = (0..nq).map(|i| series_mul(&xpow[i], &ys, k)).collect();
        for n in 0..k {
            let mut row = Vec::with_capacity(ncols);
            for xp in xpow.iter().take(np) {
                row.push(xp[n].clone());
            }
            for qs in qser.iter().take(nq) {
                row.push(qs[n].clone());
            }
            rows.push(row);
        }
    }

    let kernel = nullspace(rows, ncols);
    let mut basis = Vec::with_capacity(kernel.len());
    for v in kernel {
        let p: QPoly = trim(v[..np].to_vec());
        let q: QPoly = trim(v[np..].to_vec());
        basis.push(FunctionFieldElement::new(field.clone(), p, q, e.clone())?);
    }
    finish(d.clone(), basis, deg_d_i64, g)
}

/// Apply the Riemann–Roch self-checks, then package the answer.
fn finish(
    divisor: Divisor,
    basis: Vec<FunctionFieldElement>,
    deg_d: i64,
    g: i64,
) -> Result<RiemannRochSpace, FunctionFieldError> {
    let dim = basis.len() as i64;
    let riemann_lower = (deg_d + 1 - g).max(0);
    if dim < riemann_lower {
        return Err(FunctionFieldError::SelfCheckFailed {
            detail: format!(
                "dim L(D) = {dim} violates Riemann's inequality dim ≥ deg D + 1 − g = \
                 {riemann_lower} (deg D = {deg_d}, g = {g})"
            ),
        });
    }
    if deg_d < 0 && dim != 0 {
        return Err(FunctionFieldError::SelfCheckFailed {
            detail: format!("deg D = {deg_d} < 0 but dim L(D) = {dim} ≠ 0"),
        });
    }
    if deg_d >= 0 && dim > deg_d + 1 {
        return Err(FunctionFieldError::SelfCheckFailed {
            detail: format!("dim L(D) = {dim} exceeds deg D + 1 = {}", deg_d + 1),
        });
    }
    if deg_d > 2 * g - 2 && dim != deg_d + 1 - g {
        return Err(FunctionFieldError::SelfCheckFailed {
            detail: format!(
                "deg D = {deg_d} > 2g − 2 = {}, so Riemann–Roch forces dim L(D) = {}, but {dim} \
                 was found",
                2 * g - 2,
                deg_d + 1 - g
            ),
        });
    }
    Ok(RiemannRochSpace { divisor, basis })
}

/// `[1, xs, xs², …]` truncated to `k` terms, `n` powers.
fn power_series(xs: &[Rational], n: usize, k: usize) -> Vec<Vec<Rational>> {
    let mut out: Vec<Vec<Rational>> = Vec::with_capacity(n.max(1));
    let mut cur = vec![Rational::from(0); k];
    if k > 0 {
        cur[0] = Rational::from(1);
    }
    out.push(cur.clone());
    for _ in 1..n.max(1) {
        cur = series_mul(&cur, xs, k);
        out.push(cur.clone());
    }
    out
}

/// Power series of `x` and `y` in a local uniformiser at `place`, `k` terms.
///
/// * unramified `(α, β)`, `β ≠ 0`: `t = x − α`, so `x = α + t` and `y` is the
///   branch of `√(a(α + t))` with `y(0) = β`, from `2β·yₙ = Fₙ − Σ yᵢy_{n−i}`.
/// * branch point `(α, 0)`: `t = y`, so `y = t` and `x = α + X(t)` with
///   `a(α + X) = t²`, solved by iteration (`X ∈ t²ℚ[[t]]`, each pass gains two
///   orders, and `a'(α) ≠ 0` because `a` is squarefree).
fn local_expansions(
    a: &QPoly,
    place: &Place,
    k: usize,
) -> Result<(Vec<Rational>, Vec<Rational>), FunctionFieldError> {
    let Place::Finite { x: alpha, y: beta } = place else {
        return Err(FunctionFieldError::SelfCheckFailed {
            detail: "a local expansion was requested at infinity".into(),
        });
    };
    let shifted = taylor_shift(a, alpha);
    let f = series_of_poly(&shifted, k.max(1) + 1);

    if *beta != 0 {
        let mut xs = vec![Rational::from(0); k];
        if k > 0 {
            xs[0] = alpha.clone();
        }
        if k > 1 {
            xs[1] = Rational::from(1);
        }
        let mut ys = vec![Rational::from(0); k];
        if k > 0 {
            ys[0] = beta.clone();
        }
        let two_beta = Rational::from(2) * beta;
        for n in 1..k {
            let mut acc = f.get(n).cloned().unwrap_or_else(|| Rational::from(0));
            for i in 1..n {
                acc -= Rational::from(&ys[i] * &ys[n - i]);
            }
            ys[n] = acc / two_beta.clone();
        }
        return Ok((xs, ys));
    }

    // Branch point.  a'(α) ≠ 0 since a is squarefree.
    let g1 = eval(&poly_deriv(a), alpha);
    if g1 == 0 {
        return Err(FunctionFieldError::SelfCheckFailed {
            detail: format!(
                "a'({alpha}) = 0 at a branch point, so the curve polynomial is not squarefree"
            ),
        });
    }
    let mut xser = vec![Rational::from(0); k]; // X(t), with X ∈ t²ℚ[[t]]
    let mut t2 = vec![Rational::from(0); k];
    if k > 2 {
        t2[2] = Rational::from(1);
    }
    let deg_a = degree(a).max(0) as usize;
    // Each pass gains two orders of accuracy; k/2 + 2 passes is ample.
    for _ in 0..(k / 2 + 2) {
        let pows = power_series(&xser, deg_a + 1, k);
        let mut rhs = t2.clone();
        for (j, pj) in pows.iter().enumerate().take(deg_a + 1).skip(2) {
            let gj = shifted.get(j).cloned().unwrap_or_else(|| Rational::from(0));
            if gj == 0 {
                continue;
            }
            for (n, slot) in rhs.iter_mut().enumerate() {
                *slot -= Rational::from(&gj * &pj[n]);
            }
        }
        let next: Vec<Rational> = rhs.iter().map(|c| c.clone() / g1.clone()).collect();
        if next == xser {
            break;
        }
        xser = next;
    }
    let mut xs = xser;
    if k > 0 {
        xs[0] = alpha.clone() + xs[0].clone();
    }
    let mut ys = vec![Rational::from(0); k];
    if k > 1 {
        ys[1] = Rational::from(1);
    }
    Ok((xs, ys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;
    use crate::funcfield::field::FunctionField;
    use crate::funcfield::util::is_zero;
    use rug::Integer;

    fn e1() -> FunctionField {
        // y² = x³ − x, g = 1.
        FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap()
    }

    fn c2() -> FunctionField {
        // y² = x⁵ + 1, g = 2.
        FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 0, 0, 1]).unwrap()
    }

    fn div(f: &FunctionField, terms: &[(Place, i64)]) -> Divisor {
        Divisor::from_terms(
            f.clone(),
            terms
                .iter()
                .map(|(p, c)| (p.clone(), Integer::from(*c)))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn inf(n: i64) -> (Place, i64) {
        (Place::Infinity, n)
    }

    fn pl(x: i64, y: i64) -> Place {
        Place::finite(Rational::from(x), Rational::from(y))
    }

    #[test]
    fn dim_l_of_zero_is_one_the_constants() {
        for f in [e1(), c2()] {
            let d = Divisor::zero(f.clone()).unwrap();
            let s = riemann_roch(&d).unwrap();
            assert_eq!(s.dimension(), 1, "dim L(0) must be 1 on {f}");
            // The basis element is a constant: q = 0, p constant, e = 1.
            let u = &s.basis()[0];
            assert!(is_zero(u.numerator_algebraic_part()));
            assert_eq!(degree(u.numerator_rational_part()), 0);
        }
    }

    #[test]
    fn canonical_divisor_has_degree_2g_minus_2_and_dimension_g() {
        for f in [e1(), c2()] {
            let g = f.genus();
            let k = f.canonical_divisor().unwrap();
            assert_eq!(k.degree(), Integer::from(2 * g as i64 - 2), "deg K on {f}");
            let s = riemann_roch(&k).unwrap();
            assert_eq!(s.dimension(), g, "dim L(K) must be g on {f}");
        }
    }

    #[test]
    fn genus_zero_canonical_divisor_has_no_sections() {
        // y² = x, g = 0 ⇒ K = −2·∞, dim L(K) = 0.
        let f = FunctionField::hyperelliptic_from_i64(&[0, 1]).unwrap();
        let k = f.canonical_divisor().unwrap();
        assert_eq!(k.degree(), Integer::from(-2));
        assert_eq!(riemann_roch(&k).unwrap().dimension(), 0);
    }

    #[test]
    fn riemann_roch_equality_above_the_canonical_degree() {
        for f in [e1(), c2()] {
            let g = f.genus() as i64;
            for n in (2 * g - 1)..(2 * g + 8) {
                let d = div(&f, &[inf(n)]);
                let dim = riemann_roch(&d).unwrap().dimension() as i64;
                assert_eq!(
                    dim,
                    n + 1 - g,
                    "dim L({n}·∞) on {f} (g = {g}) must be deg D + 1 − g"
                );
            }
        }
    }

    #[test]
    fn elliptic_basis_of_l_3_infinity_is_one_x_y() {
        // L(3·∞) on y² = x³ − x is spanned by 1, x, y.
        let f = e1();
        let d = div(&f, &[inf(3)]);
        let s = riemann_roch(&d).unwrap();
        assert_eq!(s.dimension(), 3);
        // Every basis element has e = 1, deg p ≤ 1, deg q ≤ 0.
        for u in s.basis() {
            assert_eq!(degree(u.denominator()), 0);
            assert!(degree(u.numerator_rational_part()) <= 1);
            assert!(degree(u.numerator_algebraic_part()) <= 0);
        }
        // The span contains x and y: the q-parts and p-parts together span
        // {1, x} ⊕ {y}, so exactly one basis vector has a non-zero q.
        let with_y = s
            .basis()
            .iter()
            .filter(|u| !is_zero(u.numerator_algebraic_part()))
            .count();
        assert_eq!(with_y, 1);
    }

    #[test]
    fn genus_two_l_of_2g_infinity() {
        // y² = x⁵ + 1, g = 2.  L(4·∞) has dim 4 − 2 + 1 = 3 (4 = 2g > 2g−2).
        let f = c2();
        let d = div(&f, &[inf(4)]);
        let s = riemann_roch(&d).unwrap();
        assert_eq!(s.dimension(), 3);
        // Spanned by 1, x, x² — no y, since v_∞(y) = −5 < −4.
        for u in s.basis() {
            assert!(is_zero(u.numerator_algebraic_part()));
        }
    }

    #[test]
    fn genus_two_l_of_5_infinity_admits_y() {
        let f = c2();
        let d = div(&f, &[inf(5)]);
        let s = riemann_roch(&d).unwrap();
        assert_eq!(s.dimension(), 4); // 5 + 1 − 2
        assert_eq!(
            s.basis()
                .iter()
                .filter(|u| !is_zero(u.numerator_algebraic_part()))
                .count(),
            1
        );
    }

    #[test]
    fn a_single_point_on_a_positive_genus_curve_has_only_constants() {
        // deg D = 1 = 2g − 1 on g = 1 ⇒ dim = 1, and the only functions with a
        // simple pole at one point on an elliptic curve are the constants.
        let f = e1();
        for p in [Place::Infinity, pl(0, 0), pl(1, 0), pl(-1, 0)] {
            let d = div(&f, &[(p.clone(), 1)]);
            assert_eq!(
                riemann_roch(&d).unwrap().dimension(),
                1,
                "dim L({p}) on a genus-1 curve"
            );
        }
    }

    #[test]
    fn effective_divisors_at_finite_places_work_too() {
        // L(2·(0,0)) on y² = x³ − x.  deg = 2 > 2g − 2 = 0 ⇒ dim = 2.
        // (0,0) is a branch point, so v(x) = 2 there: 1 and x/1 … in fact the
        // clearing polynomial is x itself and the space is spanned by 1, ?/x.
        let f = e1();
        let d = div(&f, &[(pl(0, 0), 2)]);
        let s = riemann_roch(&d).unwrap();
        assert_eq!(s.dimension(), 2);
        // Verify each basis element really lies in L(D).
        for u in s.basis() {
            let du = u.divisor().unwrap();
            assert!(
                du.add(&d).unwrap().is_effective(),
                "basis element {u:?} is not in L(D): div(u) + D = {}",
                du.add(&d).unwrap()
            );
        }
    }

    #[test]
    fn a_negative_place_forces_a_zero_there() {
        // L(4·∞ − (0,0)) on y² = x³ − x: deg = 3 > 0 ⇒ dim = 3.
        let f = e1();
        let d = div(&f, &[inf(4), (pl(0, 0), -1)]);
        let s = riemann_roch(&d).unwrap();
        assert_eq!(s.dimension(), 3);
        for u in s.basis() {
            let du = u.divisor().unwrap();
            assert!(du.add(&d).unwrap().is_effective());
        }
    }

    #[test]
    fn negative_degree_divisors_have_no_sections() {
        let f = e1();
        for n in [-1i64, -3, -7] {
            let d = div(&f, &[inf(n)]);
            assert_eq!(riemann_roch(&d).unwrap().dimension(), 0);
        }
    }

    #[test]
    fn every_basis_element_lies_in_the_space_it_claims() {
        // The strongest check available: recompute div(u) independently and
        // confirm div(u) + D ≥ 0 for every basis element of every space.
        let f = e1();
        let cases = [
            vec![inf(3)],
            vec![inf(5)],
            vec![(pl(0, 0), 2)],
            vec![(pl(1, 0), 3)],
            vec![inf(2), (pl(-1, 0), 1)],
            vec![inf(6), (pl(0, 0), -2)],
        ];
        for terms in cases {
            let d = div(&f, &terms);
            let s = riemann_roch(&d).unwrap();
            assert!(s.dimension() > 0, "expected sections for {d}");
            for u in s.basis() {
                let du = u.divisor().unwrap();
                let sum = du.add(&d).unwrap();
                assert!(
                    sum.is_effective(),
                    "for D = {d}, basis element has div(u) = {du}, and div(u) + D = {sum} is \
                     not effective"
                );
            }
        }
    }

    #[test]
    fn the_basis_is_linearly_independent_by_construction() {
        // The nullspace basis is in "one free variable set to 1" form, so the
        // coefficient vectors are independent; check the elements differ.
        let f = c2();
        let d = div(&f, &[inf(7)]);
        let s = riemann_roch(&d).unwrap();
        assert_eq!(s.dimension(), 6);
        for i in 0..s.dimension() {
            for j in (i + 1)..s.dimension() {
                assert_ne!(s.basis()[i], s.basis()[j]);
            }
        }
    }

    #[test]
    fn real_models_refuse_riemann_roch() {
        let f = FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 0, 1]).unwrap();
        // A divisor cannot even be built there, which is the refusal.
        let err = Divisor::zero(f).unwrap_err();
        assert_eq!(err.code(), "E-FFLD-002");
    }

    #[test]
    fn local_expansion_at_a_branch_point_satisfies_the_curve() {
        let a: QPoly = [0, -1, 0, 1].iter().map(|&c| Rational::from(c)).collect();
        let k = 8;
        let (xs, ys) = local_expansions(&a, &pl(0, 0), k).unwrap();
        // y(t) = t exactly.
        assert_eq!(ys[1], Rational::from(1));
        // a(x(t)) = t².
        let pows = power_series(&xs, 4, k);
        let mut got = vec![Rational::from(0); k];
        for (i, ai) in a.iter().enumerate() {
            for (n, slot) in got.iter_mut().enumerate() {
                *slot += Rational::from(ai * &pows[i][n]);
            }
        }
        let mut want = vec![Rational::from(0); k];
        want[2] = Rational::from(1);
        assert_eq!(got, want);
    }

    #[test]
    fn local_expansion_at_an_unramified_place_satisfies_the_curve() {
        // y² = x³ + 1 at (2, 3).
        let a: QPoly = [1, 0, 0, 1].iter().map(|&c| Rational::from(c)).collect();
        let k = 6;
        let (xs, ys) = local_expansions(&a, &pl(2, 3), k).unwrap();
        assert_eq!(xs[0], Rational::from(2));
        assert_eq!(xs[1], Rational::from(1));
        assert_eq!(ys[0], Rational::from(3));
        let pows = power_series(&xs, 4, k);
        let mut got = vec![Rational::from(0); k];
        for (i, ai) in a.iter().enumerate() {
            for (n, slot) in got.iter_mut().enumerate() {
                *slot += Rational::from(ai * &pows[i][n]);
            }
        }
        assert_eq!(got, series_mul(&ys, &ys, k));
    }

    /// Independent check of the whole `L(n·∞)` ladder on `y² = x⁵ + 1` (g = 2),
    /// derived from pole orders rather than from Riemann-Roch.
    ///
    /// The functions regular away from `∞` are `ℚ[x] ⊕ ℚ[x]·y`, and at `∞` the
    /// pole orders are `v(xⁱ) = 2i` and `v(xⁱ·y) = 2i + 5`. So `dim L(n·∞)` is
    /// just a count of monomials with pole order ≤ n. That count is independent
    /// of RR, which matters most for `n ≤ 2g − 2 = 2`, where the RR *equality*
    /// says nothing and the existing tests do not reach.
    #[test]
    fn pole_order_ladder_on_a_genus_2_curve() {
        let f = c2();
        for n in 0i64..=9 {
            let expected = (0..=n / 2).count()                       // 1, x, x², …
                + if n >= 5 { (0..=(n - 5) / 2).count() } else { 0 }; // y, xy, …
            let got = riemann_roch(&div(&f, &[inf(n)])).unwrap().dimension();
            assert_eq!(got, expected, "dim L({n}·∞) on y² = x⁵ + 1");
        }
        // Spot-check the two ends against values worked out by hand.
        assert_eq!(riemann_roch(&div(&f, &[inf(0)])).unwrap().dimension(), 1);
        assert_eq!(riemann_roch(&div(&f, &[inf(2)])).unwrap().dimension(), 2);
        assert_eq!(riemann_roch(&div(&f, &[inf(5)])).unwrap().dimension(), 4);
        assert_eq!(riemann_roch(&div(&f, &[inf(7)])).unwrap().dimension(), 6);
    }
}
