//! General coupled twisted-derivation Risch DE over `ℚ(x)(α)` — Risch milestone
//! **M1-step-2**.
//!
//! Solves `D(y) + f·y = g` for `y ∈ ℚ(x)(α)`, where `g ∈ ℚ(x)(α)`, `α` is
//! algebraic of degree `d` over `ℚ(x)` given by an [`AlgExtension`] (M0), and `f`
//! is **either** a base scalar `∈ ℚ(x)` (the historical *diagonal* entry point
//! `solve_alg_rde`) **or** a general extension element `∈ ℚ(x)(α)` (the
//! *non-diagonal* coupled entry point `solve_alg_rde_general`).  This is the
//! "no new logarithm" mixed integral part: an integrand `a(x,α)·exp(kη)` whose
//! antiderivative is `v(x,α)·exp(kη)` with `v ∈ ℚ(x)(α)` and `f = kη'` — when `η`
//! is itself algebraic, `f` is non-base.
//!
//! Writing `y = Σⱼ bⱼ(x) αʲ` and substituting into `D(y) + f·y = g` collects, over
//! the power basis `{1, α, …, α^{d−1}}`, into a **coupled** first-order linear ODE
//! system `b' + M(x)·b = c`.  The pure-radical case (`αⁿ = a`) is *cyclic* and `M`
//! is diagonal — solved component-wise by
//! `super::exp_case::try_radical_poly_rde`.  This module handles the **general
//! (non-cyclic)** `α` — nested / compositum radicals such as `√x + √(x+1)` — where
//! `M` genuinely couples the components.
//!
//! ## Method
//!
//! An undetermined-coefficient ansatz `bⱼ = pⱼ(x)/Den(x)` over candidate
//! denominators `Den` and bounded numerator degree.  Because the operator
//! `L(y) = D(y) + f·y` is `ℚ`-linear in the unknown coefficients of the `pⱼ`, we
//! evaluate `L` on each basis element `αʲ·xᵐ/Den`, clear the common
//! `x`-denominator of every power-basis component (then match each `xᵐ`), and
//! assemble an exact `ℚ`-linear system `A·u = c`.  We Gauss-solve it and **verify
//! `D(y) + f·y = g` exactly in the field** before returning.
//!
//! This is *sound by construction*: the linear system is faithful to `L(y) = g`
//! (denominator clearing is exact and every `x`-power is matched), and the final
//! field equality is an independent check — a denominator/degree bound too small
//! to contain the true solution yields `None` (incomplete), never a wrong
//! antiderivative.

use rug::Rational;

use super::alg_field::{AlgElem, AlgExtension, RatFn};
use super::poly_rde::{poly_mul, poly_one, trim, QPoly};
use super::rational_rde::{poly_div_exact, poly_gcd};

/// Max numerator degree tried in the ansatz `bⱼ = pⱼ(x)/Den`.
const DEG_CAP: usize = 6;

/// Solve `D(y) + f·y = g` for `y ∈ ℚ(x)(α)` with a **base scalar** `f ∈ ℚ(x)`,
/// or `None` if no rational solution is found within the ansatz bounds.  Sound by
/// exact verification — see the module docs.
///
/// Thin wrapper over [`solve_alg_rde_general`] that lifts the base scalar `f` to
/// the constant extension element `e.constant(f)`; it preserves the exact
/// behavior of the historical base-`f` solver for existing callers.
pub(crate) fn solve_alg_rde(e: &AlgExtension, f: &RatFn, g: &AlgElem) -> Option<AlgElem> {
    solve_alg_rde_general(e, &e.constant(f.clone()), g)
}

/// Solve `D(y) + f·y = g` for `y ∈ ℚ(x)(α)` with a **general** `f ∈ ℚ(x)(α)`
/// (possibly carrying `y`-powers — the *non-diagonal* coupled case), or `None`
/// if no rational solution is found within the ansatz bounds.
///
/// The operator `L(y) = D(y) + f·y` is `ℚ`-linear in the unknown coefficients of
/// the ansatz `y = Σⱼ (pⱼ(x)/Den) αʲ` regardless of whether `f` is a base scalar
/// or a general extension element — only the multiplication `f·(·)` and the
/// denominator collection differ from the base-`f` path.  As in the base case we
/// assemble an exact `ℚ`-linear system, Gauss-solve it, and **verify
/// `D(y)+f·y=g` exactly in the field** before returning, so the result is sound
/// for general `f` automatically.
pub(crate) fn solve_alg_rde_general(e: &AlgExtension, f: &AlgElem, g: &AlgElem) -> Option<AlgElem> {
    let d = e.degree() as usize;
    if d == 0 {
        return None;
    }
    let dens = candidate_denominators(e, f, g, d);
    for den in &dens {
        for ncap in 0..=DEG_CAP {
            if let Some(y) = solve_with_denominator(e, f, g, den, ncap, d) {
                return Some(y);
            }
        }
    }
    None
}

/// Candidate `x`-denominators for `y`, increasing in complexity: `1`, then the
/// LCM `B` of every `x`-denominator that appears in `D(αʲ)`, `f`, and `g`, then
/// `B²`, `B³`.  Over-clearing is harmless (the numerator ansatz just needs more
/// terms); verification guards correctness.
fn candidate_denominators(e: &AlgExtension, f: &AlgElem, g: &AlgElem, d: usize) -> Vec<QPoly> {
    let mut base = poly_one();
    let gen = e.generator();
    for j in 0..d {
        if let Some(aj) = e.pow(&gen, j as i64) {
            for c in e.derivation(&aj) {
                base = poly_lcm(&base, c.denom());
            }
        }
    }
    // f may be a general extension element — LCM every component's denominator.
    for c in f {
        base = poly_lcm(&base, c.denom());
    }
    for c in g {
        base = poly_lcm(&base, c.denom());
    }
    let base2 = poly_mul(&base, &base);
    let base3 = poly_mul(&base2, &base);
    let mut out = vec![poly_one(), base, base2, base3];
    out.dedup_by(|a, b| trim(a.clone()) == trim(b.clone()));
    out
}

/// Solve seeking `y = Σⱼ (pⱼ(x)/Den) αʲ` with `deg pⱼ ≤ ncap`, for the fixed `Den`.
fn solve_with_denominator(
    e: &AlgExtension,
    f: &AlgElem,
    g: &AlgElem,
    den: &QPoly,
    ncap: usize,
    d: usize,
) -> Option<AlgElem> {
    // Ansatz basis: component j carries numerator xᵐ over the common Den.
    let basis: Vec<(usize, usize)> = (0..d)
        .flat_map(|j| (0..=ncap).map(move |m| (j, m)))
        .collect();
    let elems: Vec<AlgElem> = basis
        .iter()
        .map(|&(j, m)| {
            let coeff = RatFn::new(x_pow(m), den.clone()); // xᵐ / Den
            let mut v = vec![RatFn::int(0); d];
            v[j] = coeff;
            e.reduce(&v)
        })
        .collect();

    // L(·) = D(·) + f·(·) applied to each basis element.  `f` is a general
    // extension element, so `e.mul(f, m)` mixes the power basis (the coupling).
    let cols: Vec<AlgElem> = elems
        .iter()
        .map(|m| e.add(&e.derivation(m), &e.mul(f, m)))
        .collect();

    let (matrix, rhs) = extract_linear_system(&cols, g, d);
    let sol = gauss_solve(matrix, rhs, basis.len())?;

    // Reconstruct y = Σ solᵢ · elemᵢ.
    let mut y = e.from_int(0);
    for (idx, elem) in elems.iter().enumerate() {
        if sol[idx] != 0 {
            let s = e.constant(RatFn::from_poly(&vec![sol[idx].clone()]));
            y = e.add(&y, &e.mul(&s, elem));
        }
    }

    // Exact verification: D(y) + f·y == g.
    let lhs = e.add(&e.derivation(&y), &e.mul(f, &y));
    if e.elem_eq(&lhs, g) {
        Some(y)
    } else {
        None
    }
}

/// Build the exact `ℚ`-linear system `Σᵢ uᵢ·colᵢ = target` by, for each power-basis
/// component `k`, clearing the common `x`-denominator of the `ℚ(x)` entries and
/// matching every `xᵐ` coefficient.
fn extract_linear_system(
    cols: &[AlgElem],
    target: &AlgElem,
    d: usize,
) -> (Vec<Vec<Rational>>, Vec<Rational>) {
    let comp =
        |a: &AlgElem, k: usize| -> RatFn { a.get(k).cloned().unwrap_or_else(|| RatFn::int(0)) };
    let mut matrix: Vec<Vec<Rational>> = Vec::new();
    let mut rhs: Vec<Rational> = Vec::new();

    for k in 0..d {
        let col_rf: Vec<RatFn> = cols.iter().map(|c| comp(c, k)).collect();
        let tgt_rf = comp(target, k);

        // Common x-denominator of this component across all columns and target.
        let mut d_x = poly_one();
        for r in &col_rf {
            d_x = poly_lcm(&d_x, r.denom());
        }
        d_x = poly_lcm(&d_x, tgt_rf.denom());

        let s_cols: Vec<QPoly> = col_rf
            .iter()
            .map(|r| poly_mul(r.numer(), &poly_div_exact(&d_x, r.denom())))
            .collect();
        let s_tgt = poly_mul(tgt_rf.numer(), &poly_div_exact(&d_x, tgt_rf.denom()));

        let max_m = s_cols
            .iter()
            .map(|s| s.len())
            .chain(std::iter::once(s_tgt.len()))
            .max()
            .unwrap_or(0);
        for m in 0..max_m {
            matrix.push(
                s_cols
                    .iter()
                    .map(|s| s.get(m).cloned().unwrap_or_else(|| Rational::from(0)))
                    .collect(),
            );
            rhs.push(s_tgt.get(m).cloned().unwrap_or_else(|| Rational::from(0)));
        }
    }
    (matrix, rhs)
}

/// Solve `M·x = b` over `ℚ` by Gauss–Jordan, returning a particular solution
/// (free variables set to 0) or `None` if inconsistent.
fn gauss_solve(
    mut m: Vec<Vec<Rational>>,
    mut b: Vec<Rational>,
    ncols: usize,
) -> Option<Vec<Rational>> {
    let nrows = m.len();
    let mut pivot_row_of_col: Vec<Option<usize>> = vec![None; ncols];
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
            *v = v.clone() / piv.clone();
        }
        b[row] = b[row].clone() / piv.clone();
        let pivot_row = m[row].clone();
        let pivot_b = b[row].clone();
        for r in 0..nrows {
            if r != row && m[r][col] != 0 {
                let factor = m[r][col].clone();
                for (dst, pv) in m[r].iter_mut().zip(pivot_row.iter()) {
                    *dst -= factor.clone() * pv.clone();
                }
                b[r] -= factor * pivot_b.clone();
            }
        }
        pivot_row_of_col[col] = Some(row);
        row += 1;
    }
    for r in 0..nrows {
        if m[r].iter().all(|v| *v == 0) && b[r] != 0 {
            return None;
        }
    }
    let mut x = vec![Rational::from(0); ncols];
    for (col, pr) in pivot_row_of_col.iter().enumerate() {
        if let Some(r) = pr {
            x[col] = b[*r].clone();
        }
    }
    Some(x)
}

/// `lcm(a, b)` over `ℚ[x]` (non-monic is fine — used only as a clearing factor).
fn poly_lcm(a: &QPoly, b: &QPoly) -> QPoly {
    let g = poly_gcd(a, b);
    poly_div_exact(&poly_mul(a, b), &g)
}

/// The monomial `xᵐ` as a `ℚ[x]` polynomial.
fn x_pow(m: usize) -> QPoly {
    let mut p = vec![Rational::from(0); m + 1];
    p[m] = Rational::from(1);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::risch::poly_rde::poly_scale;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }

    /// Cyclic sanity check: `α = √x` (`α² = x`), solve `D(y) = (3/2)·α`.  The
    /// antiderivative is `y = x·α = x^{3/2}` since `D(x^{3/2}) = (3/2)x^{1/2}`.
    #[test]
    fn cyclic_sqrt_recovers_solution() {
        let e = AlgExtension::radical(2, &vec![rat(0), rat(1)]); // α² = x
                                                                 // g = (3/2)·α  (= component vector [0, 3/2]).
        let g: AlgElem = vec![
            RatFn::int(0),
            RatFn::from_poly(&vec![Rational::from((3, 2))]),
        ];
        let f = RatFn::int(0);
        let y = solve_alg_rde(&e, &f, &g).expect("should solve");
        // D(y) must equal g.
        assert!(e.elem_eq(&e.derivation(&y), &g));
        // y = x·α.
        let expected: AlgElem = vec![RatFn::int(0), RatFn::from_poly(&vec![rat(0), rat(1)])];
        assert!(e.elem_eq(&y, &expected), "y = {:?}", y);
    }

    /// **Non-cyclic** (the M1-step-2 case): `α = √x + √(x+1)`, a degree-4
    /// extension whose minimal polynomial `α⁴ − 2(2x+1)α² + 1 = 0` is *not* a
    /// pure radical, so `D(α)` mixes the power basis and the system is coupled.
    /// We construct `g = D(α)` and confirm the solver recovers a `y` with
    /// `D(y) = g` (namely `y = α`).
    #[test]
    fn noncyclic_compositum_pure_antiderivative() {
        // q = α⁴ − 2(2x+1)α² + 1 : coeffs ascending [1, 0, −2(2x+1), 0, 1].
        let q: Vec<QPoly> = vec![
            poly_one(),                                  // 1
            Vec::new(),                                  // 0·α
            poly_scale(&vec![rat(1), rat(2)], &rat(-2)), // −2(2x+1) = −4x−2
            Vec::new(),                                  // 0·α³
            poly_one(),                                  // α⁴
        ];
        let e = AlgExtension::new(&q);
        assert_eq!(e.degree(), 4);

        let alpha = e.generator();
        let g = e.derivation(&alpha); // genuinely coupled element
        let f = RatFn::int(0);
        let y = solve_alg_rde(&e, &f, &g).expect("coupled RDE should solve");
        assert!(
            e.elem_eq(&e.derivation(&y), &g),
            "D(y) must equal g; y = {y:?}"
        );
    }

    /// Non-cyclic with a nonzero `f`: with `α = √x + √(x+1)` and `f = 1/x`,
    /// `g = D(α) + (1/x)·α` is solved by `y = α`.
    #[test]
    fn noncyclic_compositum_with_f() {
        let q: Vec<QPoly> = vec![
            poly_one(),
            Vec::new(),
            poly_scale(&vec![rat(1), rat(2)], &rat(-2)),
            Vec::new(),
            poly_one(),
        ];
        let e = AlgExtension::new(&q);
        let alpha = e.generator();
        let f = RatFn::new(poly_one(), vec![rat(0), rat(1)]); // 1/x
        let f_elem = e.constant(f.clone());
        let g = e.add(&e.derivation(&alpha), &e.mul(&f_elem, &alpha));
        let y = solve_alg_rde(&e, &f, &g).expect("coupled RDE with f should solve");
        let lhs = e.add(&e.derivation(&y), &e.mul(&f_elem, &y));
        assert!(e.elem_eq(&lhs, &g), "D(y)+f·y must equal g; y = {y:?}");
    }

    /// A target with **no** rational solution must return `None` (never a wrong
    /// antiderivative).  `g = 1/x` (embedded from `ℚ(x)`) has antiderivative
    /// `log x ∉ ℚ(x)(α)`, so no rational `y` solves `D(y) = 1/x`.
    #[test]
    fn unsolvable_log_returns_none() {
        let q: Vec<QPoly> = vec![
            poly_one(),
            Vec::new(),
            poly_scale(&vec![rat(1), rat(2)], &rat(-2)),
            Vec::new(),
            poly_one(),
        ];
        let e = AlgExtension::new(&q);
        // g = 1/x (constant element of ℚ(x)).
        let g = e.constant(RatFn::new(poly_one(), vec![rat(0), rat(1)]));
        let f = RatFn::int(0);
        assert!(solve_alg_rde(&e, &f, &g).is_none());
    }

    // -- Non-base (non-diagonal) f: the genuine coupled case --------------

    /// **Non-base `f` over `α = √x`.**  Take `f = 1/(2√x) = (1/(2x))·α` (a
    /// non-base extension element, the `∫exp(√x)` twist `D(√x)`) and `y = √x`.
    /// Then `g = D(y) + f·y = (1/(2x))·α + (1/(2x))·α·α = (1/(2x))·α + 1/2`.
    /// The generalized solver must recover a `y'` with `D(y')+f·y' = g`.
    #[test]
    fn nonbase_f_sqrt_coupled() {
        let e = AlgExtension::radical(2, &vec![rat(0), rat(1)]); // α² = x
        let alpha = e.generator();
        // f = (1/(2x))·α   (= 1/(2√x))
        let f: AlgElem = vec![RatFn::int(0), RatFn::new(poly_one(), vec![rat(0), rat(2)])];
        // y = √x = α
        let y_true = alpha.clone();
        let g = e.add(&e.derivation(&y_true), &e.mul(&f, &y_true));
        let y = solve_alg_rde_general(&e, &f, &g).expect("non-base f coupled solve");
        let lhs = e.add(&e.derivation(&y), &e.mul(&f, &y));
        assert!(e.elem_eq(&lhs, &g), "D(y)+f·y must equal g; y = {y:?}");
        // And the headline antiderivative is recovered.
        assert!(e.elem_eq(&y, &y_true), "expected y = √x; got {y:?}");
    }

    /// **Non-base `f` over `α = ∛x`.**  `f = (1/x)·α²` (a non-base element) and
    /// `y = x` (a base function, but the equation couples via `f`).
    /// `g = D(y) + f·y = 1 + (1/x)·α²·x = 1 + α²`.  Solver must recover it.
    #[test]
    fn nonbase_f_cbrt_coupled() {
        let e = AlgExtension::radical(3, &vec![rat(0), rat(1)]); // α³ = x
                                                                 // f = (1/x)·α²
        let f: AlgElem = vec![
            RatFn::int(0),
            RatFn::int(0),
            RatFn::new(poly_one(), vec![rat(0), rat(1)]),
        ];
        // y = x
        let y_true = e.constant(RatFn::from_poly(&vec![rat(0), rat(1)]));
        let g = e.add(&e.derivation(&y_true), &e.mul(&f, &y_true));
        let y = solve_alg_rde_general(&e, &f, &g).expect("non-base cbrt f coupled solve");
        let lhs = e.add(&e.derivation(&y), &e.mul(&f, &y));
        assert!(e.elem_eq(&lhs, &g), "D(y)+f·y must equal g; y = {y:?}");
        assert!(e.elem_eq(&y, &y_true), "expected y = x; got {y:?}");
    }

    /// **Wrapper equivalence / regression.**  For a base scalar `f`, the thin
    /// `solve_alg_rde(e, &f, g)` wrapper must return exactly what
    /// `solve_alg_rde_general(e, &e.constant(f), g)` returns.
    #[test]
    fn base_f_wrapper_matches_general() {
        let e = AlgExtension::radical(2, &vec![rat(0), rat(1)]); // α² = x
        let alpha = e.generator();
        let f = RatFn::new(poly_one(), vec![rat(0), rat(1)]); // 1/x ∈ base
        let f_elem = e.constant(f.clone());
        let g = e.add(&e.derivation(&alpha), &e.mul(&f_elem, &alpha));
        let via_wrapper = solve_alg_rde(&e, &f, &g);
        let via_general = solve_alg_rde_general(&e, &f_elem, &g);
        match (&via_wrapper, &via_general) {
            (Some(a), Some(b)) => assert!(e.elem_eq(a, b), "wrapper vs general differ"),
            (None, None) => {}
            _ => panic!("wrapper vs general disagree on solvability"),
        }
        assert!(via_wrapper.is_some(), "base-f case should solve");
    }

    /// **Unsolvable non-base `f`.**  With `f = α` (non-base) and `g = 1/x`
    /// (whose antiderivative would be `log x ∉ ℚ(x)(α)`), there is no rational
    /// `y` — the generalized solver must return `None`, never a wrong answer.
    #[test]
    fn nonbase_f_unsolvable_returns_none() {
        let e = AlgExtension::radical(2, &vec![rat(0), rat(1)]); // α² = x
        let f = e.generator(); // α (non-base)
        let g = e.constant(RatFn::new(poly_one(), vec![rat(0), rat(1)])); // 1/x
        assert!(solve_alg_rde_general(&e, &f, &g).is_none());
    }
}
