//! Exponential tower differential field `ℚ(x)(t)`, `t = exp(η)`, as a
//! [`CoeffField`] — the Risch **MD** substrate for radicands that involve the
//! transcendental (e.g. `∛(x + eˣ)`, tutorial Example 15).
//!
//! An element is a rational function in `t` with `ℚ(x)`-coefficients
//! ([`RatFn`]), i.e. a fraction of polynomials in `t` over `ℚ(x)`.  The
//! polynomial-in-`t` arithmetic reuses the generic [`CoeffField`] machinery
//! instantiated at [`RationalFunctionField`] (the `ℚ(x)` coefficients), and the
//! derivation is the tower derivation
//!
//! ```text
//!   D(t) = η'·t,   D(Σⱼ cⱼ(x) tʲ) = Σⱼ (cⱼ'(x) + j·η'·cⱼ(x)) tʲ
//! ```
//!
//! Because `ℚ(x)(t)` is itself a [`CoeffField`], the generic `Quotient` gives
//! an algebraic extension `ℚ(x)(t)[y]/(q(x, t, y))` for free, and
//! `radical_dy` computes `D(y) = −q_x/q_y` in the tower.  This is exactly the
//! M0 "substitute a transcendental tower for the coefficient field" hook: the
//! radicand may now involve `t`.
//!
//! This module provides the differential-algebra substrate (arithmetic +
//! derivation) **and** [`solve_tower_rde`], the per-component Risch DE solver
//! `vᵢ' + ωᵢ vᵢ = cᵢ` over the tower (verification-guarded, allowing `vᵢ` with
//! denominators).  The end-to-end integrator that drives it lives in
//! [`super::tower_integrate`].

use rug::Rational;

use super::alg_field::{RatFn, RationalFunctionField};
use super::number_field::{
    gdegree, gext_gcd, gpoly_add, gpoly_divrem, gpoly_mul, gpoly_scale, gtrim, CoeffField, GPoly,
};
use super::poly_rde::{poly_mul, QPoly};
use super::rational_rde::{poly_div_exact, poly_gcd};

/// A polynomial in `t` over `ℚ(x)` (ascending degree in `t`).
type TPoly = GPoly<RationalFunctionField>;

fn qx() -> RationalFunctionField {
    RationalFunctionField
}

/// An element of `ℚ(x)(t)`: a canonical fraction `num/den` of `t`-polynomials
/// over `ℚ(x)` (coprime, monic `den`, `0 = ⟨⟩/1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TExpr {
    num: TPoly,
    den: TPoly,
}

impl TExpr {
    /// Build `num/den` in canonical form.  Panics if `den` is the zero
    /// polynomial.
    pub fn new(num: TPoly, den: TPoly) -> Self {
        let f = qx();
        let num = gtrim(&f, num);
        let den = gtrim(&f, den);
        assert!(!den.is_empty(), "TExpr: zero denominator");
        if num.is_empty() {
            return Self {
                num: Vec::new(),
                den: vec![f.one()],
            };
        }
        // Reduce by gcd over ℚ(x)[t].
        let (g, _, _) = gext_gcd(&f, &num, &den);
        let num = gpoly_divrem(&f, &num, &g).0;
        let den = gpoly_divrem(&f, &den, &g).0;
        // Normalize: make the denominator monic in t.
        let lead = den[gdegree(&f, &den) as usize].clone();
        let lead_inv = f
            .inv(&lead)
            .expect("nonzero ℚ(x) leading coeff is invertible");
        let num = gpoly_scale(&f, &num, &lead_inv);
        let den = gpoly_scale(&f, &den, &lead_inv);
        Self { num, den }
    }

    /// The constant (in `t`) element `r ∈ ℚ(x)`.
    pub fn from_ratfn(r: RatFn) -> Self {
        Self::new(vec![r], vec![qx().one()])
    }

    /// The monomial `t` (the exponential generator itself).
    pub fn t() -> Self {
        Self::new(vec![qx().zero(), qx().one()], vec![qx().one()])
    }

    /// Integer constant `n`.
    pub fn int(n: i64) -> Self {
        Self::from_ratfn(RatFn::int(n))
    }

    /// Numerator (canonical, polynomial in `t`).
    pub fn numer(&self) -> &TPoly {
        &self.num
    }
    /// Denominator (canonical, monic polynomial in `t`).
    pub fn denom(&self) -> &TPoly {
        &self.den
    }

    fn is_zero(&self) -> bool {
        self.num.is_empty()
    }

    fn add(&self, other: &Self) -> Self {
        let f = qx();
        let num = gpoly_add(
            &f,
            &gpoly_mul(&f, &self.num, &other.den),
            &gpoly_mul(&f, &other.num, &self.den),
        );
        Self::new(num, gpoly_mul(&f, &self.den, &other.den))
    }

    fn mul(&self, other: &Self) -> Self {
        let f = qx();
        Self::new(
            gpoly_mul(&f, &self.num, &other.num),
            gpoly_mul(&f, &self.den, &other.den),
        )
    }

    fn neg(&self) -> Self {
        let f = qx();
        let neg1 = f.neg(&f.one());
        Self::new(gpoly_scale(&f, &self.num, &neg1), self.den.clone())
    }

    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(Self::new(self.den.clone(), self.num.clone()))
        }
    }
}

/// `D(P)` for a `t`-polynomial `P = Σⱼ cⱼ tʲ` in an **exponential** tower
/// (`Dt = η'·t`): `Σⱼ (cⱼ' + j·η'·cⱼ) tʲ` — the `t`-degree is preserved.
fn exp_tpoly_derivation(p: &TPoly, deta: &RatFn) -> TPoly {
    let f = qx();
    let mut out: TPoly = Vec::with_capacity(p.len());
    for (j, cj) in p.iter().enumerate() {
        let dc = f.derivation(cj); // cⱼ'
        let drift = f.mul(&f.mul(&RatFn::int(j as i64), deta), cj); // j·η'·cⱼ
        out.push(f.add(&dc, &drift));
    }
    gtrim(&f, out)
}

/// `D(P)` for a `t`-polynomial `P = Σⱼ cⱼ tʲ` in a **logarithmic** tower
/// (`Dt = h'/h`, free of `t`): `D(cⱼ tʲ) = cⱼ' tʲ + j·(h'/h)·cⱼ·t^{j−1}`, so
/// collecting at degree `p` gives `cₚ' + (p+1)·(h'/h)·cₚ₊₁` — the `t`-degree is
/// *lowered* by one in the drift term.
fn log_tpoly_derivation(p: &TPoly, dh_over_h: &RatFn) -> TPoly {
    let f = qx();
    let mut out: TPoly = (0..p.len()).map(|_| f.zero()).collect();
    for (j, cj) in p.iter().enumerate() {
        out[j] = f.add(&out[j], &f.derivation(cj)); // cⱼ' at degree j
        if j >= 1 {
            // j·(h'/h)·cⱼ at degree j−1
            let term = f.mul(&f.mul(&RatFn::int(j as i64), dh_over_h), cj);
            out[j - 1] = f.add(&out[j - 1], &term);
        }
    }
    gtrim(&f, out)
}

/// Apply the quotient rule given the `t`-derivatives of numerator/denominator:
/// `D(num/den) = (D(num)·den − num·D(den)) / den²`.
fn texpr_from_quotient_rule(num: &TPoly, den: &TPoly, dnum: &TPoly, dden: &TPoly) -> TExpr {
    let f = qx();
    let lhs = gpoly_mul(&f, dnum, den);
    let rhs = gpoly_mul(&f, num, dden);
    let neg1 = f.neg(&f.one());
    let numer = gpoly_add(&f, &lhs, &gpoly_scale(&f, &rhs, &neg1));
    TExpr::new(numer, gpoly_mul(&f, den, den))
}

/// The exponential tower `ℚ(x)(t)`, `t = exp(η)`, parameterized by `η' = deta`.
#[derive(Clone, Debug)]
pub struct ExpTowerField {
    /// `η'(x)` — the derivative of the exponent, a `ℚ(x)` element.
    pub deta: RatFn,
}

impl ExpTowerField {
    pub fn new(deta: RatFn) -> Self {
        Self { deta }
    }
}

impl CoeffField for ExpTowerField {
    type Elem = TExpr;

    fn zero(&self) -> TExpr {
        TExpr::int(0)
    }
    fn one(&self) -> TExpr {
        TExpr::int(1)
    }
    fn from_i64(&self, n: i64) -> TExpr {
        TExpr::int(n)
    }
    fn add(&self, a: &TExpr, b: &TExpr) -> TExpr {
        a.add(b)
    }
    fn sub(&self, a: &TExpr, b: &TExpr) -> TExpr {
        a.add(&b.neg())
    }
    fn mul(&self, a: &TExpr, b: &TExpr) -> TExpr {
        a.mul(b)
    }
    fn neg(&self, a: &TExpr) -> TExpr {
        a.neg()
    }
    fn inv(&self, a: &TExpr) -> Option<TExpr> {
        a.inv()
    }
    fn is_zero(&self, a: &TExpr) -> bool {
        a.is_zero()
    }
    fn eq(&self, a: &TExpr, b: &TExpr) -> bool {
        a == b
    }

    /// `D(num/den)`, with the exponential-tower `t`-derivation.
    fn derivation(&self, a: &TExpr) -> TExpr {
        let dnum = exp_tpoly_derivation(&a.num, &self.deta);
        let dden = exp_tpoly_derivation(&a.den, &self.deta);
        texpr_from_quotient_rule(&a.num, &a.den, &dnum, &dden)
    }
}

/// The logarithmic tower `ℚ(x)(t)`, `t = log(h)`, parameterized by `h'/h`.
///
/// Same element type and arithmetic as [`ExpTowerField`]; only the derivation
/// differs (`Dt = h'/h`, which lowers the `t`-degree in the drift term).
#[derive(Clone, Debug)]
pub struct LogTowerField {
    /// `h'/h` — the logarithmic derivative of the inner argument, a `ℚ(x)` element.
    pub dh_over_h: RatFn,
}

impl LogTowerField {
    pub fn new(dh_over_h: RatFn) -> Self {
        Self { dh_over_h }
    }
}

impl CoeffField for LogTowerField {
    type Elem = TExpr;

    fn zero(&self) -> TExpr {
        TExpr::int(0)
    }
    fn one(&self) -> TExpr {
        TExpr::int(1)
    }
    fn from_i64(&self, n: i64) -> TExpr {
        TExpr::int(n)
    }
    fn add(&self, a: &TExpr, b: &TExpr) -> TExpr {
        a.add(b)
    }
    fn sub(&self, a: &TExpr, b: &TExpr) -> TExpr {
        a.add(&b.neg())
    }
    fn mul(&self, a: &TExpr, b: &TExpr) -> TExpr {
        a.mul(b)
    }
    fn neg(&self, a: &TExpr) -> TExpr {
        a.neg()
    }
    fn inv(&self, a: &TExpr) -> Option<TExpr> {
        a.inv()
    }
    fn is_zero(&self, a: &TExpr) -> bool {
        a.is_zero()
    }
    fn eq(&self, a: &TExpr, b: &TExpr) -> bool {
        a == b
    }

    /// `D(num/den)`, with the logarithmic-tower `t`-derivation.
    fn derivation(&self, a: &TExpr) -> TExpr {
        let dnum = log_tpoly_derivation(&a.num, &self.dh_over_h);
        let dden = log_tpoly_derivation(&a.den, &self.dh_over_h);
        texpr_from_quotient_rule(&a.num, &a.den, &dnum, &dden)
    }
}

// ===========================================================================
// Risch differential equation solver over the tower ℚ(x)(t)  (MD step)
// ===========================================================================

/// LCM of two `t`-polynomials over `ℚ(x)`.
fn tpoly_lcm(a: &TPoly, b: &TPoly) -> TPoly {
    let f = qx();
    if a.is_empty() {
        return b.clone();
    }
    if b.is_empty() {
        return a.clone();
    }
    let (g, _, _) = gext_gcd(&f, a, b);
    gpoly_divrem(&f, &gpoly_mul(&f, a, b), &g).0
}

/// LCM of two `ℚ[x]` polynomials.
fn poly_lcm(a: &QPoly, b: &QPoly) -> QPoly {
    if a.iter().all(|c| *c == 0) {
        return b.clone();
    }
    if b.iter().all(|c| *c == 0) {
        return a.clone();
    }
    poly_div_exact(&poly_mul(a, b), &poly_gcd(a, b))
}

/// The monomial `coeff·xᵏ·tʲ` as a tower element.
fn monomial(j: usize, k: usize, coeff: &Rational) -> TExpr {
    let mut xk = vec![Rational::from(0); k + 1];
    xk[k] = coeff.clone();
    let rk = RatFn::new(xk, vec![Rational::from(1)]);
    let mut num: TPoly = vec![RatFn::int(0); j];
    num.push(rk);
    TExpr::new(num, vec![RatFn::int(1)])
}

/// Solve `v' + ω·v = c` for `v ∈ ℚ(x)(t)` by an undetermined-coefficient ansatz
/// `v = (Σⱼₖ cⱼₖ xᵏ tʲ) / D`, trying a sequence of candidate denominators `D`
/// derived from `ω` and `c` (starting with `D = 1`, the polynomial case).
///
/// **Sound by construction:** each candidate's `v` is verified to satisfy
/// `D(v) + ω·v = c` *exactly* in the tower field before being returned, so a
/// `Some` is always correct.  `None` means "no solution `U/D` was found for any
/// tried `D` within the degree caps" — it is **not** a non-elementarity
/// certificate.
pub fn solve_tower_rde<F: CoeffField<Elem = TExpr>>(
    field: &F,
    omega: &TExpr,
    c: &TExpr,
) -> Option<TExpr> {
    candidate_denominators(field, omega, c)
        .iter()
        .find_map(|d| solve_with_denominator(field, omega, c, d))
}

/// Candidate denominators for `v`, in increasing complexity: `1`, the full
/// denominators of `c` and `ω`, and a few products/powers.  Over-clearing is
/// harmless (the numerator ansatz just needs more terms); verification guards
/// correctness.
fn candidate_denominators<F: CoeffField<Elem = TExpr>>(
    field: &F,
    omega: &TExpr,
    c: &TExpr,
) -> Vec<TExpr> {
    let one = field.one();
    let dc = full_denominator(c);
    let dw = full_denominator(omega);
    let dcw = field.mul(&dc, &dw);
    let dc2 = field.mul(&dc, &dc);
    let mut cands = vec![
        one,
        dc.clone(),
        dw.clone(),
        dcw.clone(),
        dc2.clone(),
        field.mul(&dc2, &dw),
        field.mul(&dcw, &dw),
    ];
    cands.dedup_by(|a, b| a == b);
    cands
}

/// The denominator that clears `e` to a polynomial in `(x, t)`: the `t`-poly
/// denominator times the LCM of the `x`-denominators of every coefficient.
fn full_denominator(e: &TExpr) -> TExpr {
    let mut d_x = vec![Rational::from(1)];
    for rf in e.numer().iter().chain(e.denom().iter()) {
        d_x = poly_lcm(&d_x, rf.denom());
    }
    let t_den = TExpr::new(e.denom().clone(), vec![RatFn::int(1)]);
    let x_den = TExpr::from_ratfn(RatFn::from_poly(&d_x));
    let f = qx();
    // t_den · x_den as a tower element (both have trivial denominators).
    TExpr::new(
        gpoly_mul(&f, t_den.numer(), x_den.numer()),
        vec![RatFn::int(1)],
    )
}

/// Solve `v' + ω·v = c` seeking `v = (Σⱼₖ cⱼₖ xᵏ tʲ)/D` for the fixed `D`.
fn solve_with_denominator<F: CoeffField<Elem = TExpr>>(
    field: &F,
    omega: &TExpr,
    c: &TExpr,
    d: &TExpr,
) -> Option<TExpr> {
    const JCAP: usize = 3; // max degree in t
    const KCAP: usize = 5; // max degree in x

    let inv_d = field.inv(d)?;
    let basis: Vec<(usize, usize)> = (0..=JCAP)
        .flat_map(|j| (0..=KCAP).map(move |k| (j, k)))
        .collect();
    // Basis element xᵏtʲ/D, and its image L(·) = D(·) + ω·(·).
    let one = Rational::from(1);
    let elems: Vec<TExpr> = basis
        .iter()
        .map(|&(j, k)| field.mul(&monomial(j, k, &one), &inv_d))
        .collect();
    let cols: Vec<TExpr> = elems
        .iter()
        .map(|m| field.add(&field.derivation(m), &field.mul(omega, m)))
        .collect();

    let (matrix, rhs) = extract_linear_system(&cols, c);
    let sol = gauss_solve(matrix, rhs, basis.len())?;

    // Reconstruct v = Σ solⱼₖ (xᵏ tʲ / D).
    let mut v = field.zero();
    for (idx, elem) in elems.iter().enumerate() {
        if sol[idx] != 0 {
            v = field.add(
                &v,
                &field.mul(
                    &TExpr::from_ratfn(RatFn::from_poly(&vec![sol[idx].clone()])),
                    elem,
                ),
            );
        }
    }

    // Exact verification: D(v) + ω·v == c.
    let lhs = field.add(&field.derivation(&v), &field.mul(omega, &v));
    if field.eq(&lhs, c) {
        Some(v)
    } else {
        None
    }
}

/// Build the exact ℚ-linear system `Σⱼₖ cⱼₖ·colⱼₖ = target` by clearing the
/// common `t`-denominator (→ match each `tᵖ`) then the common `x`-denominator of
/// each resulting `ℚ(x)` equation (→ match each `xᵐ`).
fn extract_linear_system(cols: &[TExpr], target: &TExpr) -> (Vec<Vec<Rational>>, Vec<Rational>) {
    let f = qx();
    // Common t-denominator over all columns and the target.
    let mut d_t = vec![f.one()];
    for col in cols {
        d_t = tpoly_lcm(&d_t, &col.den);
    }
    d_t = tpoly_lcm(&d_t, &target.den);

    let scale = |e: &TExpr| -> TPoly {
        let factor = gpoly_divrem(&f, &d_t, &e.den).0; // d_t / e.den (exact)
        gpoly_mul(&f, &e.num, &factor)
    };
    let n_cols: Vec<TPoly> = cols.iter().map(scale).collect();
    let n_target = scale(target);

    let max_p = n_cols
        .iter()
        .map(|n| n.len())
        .chain(std::iter::once(n_target.len()))
        .max()
        .unwrap_or(0);

    let coeff_t =
        |n: &TPoly, p: usize| -> RatFn { n.get(p).cloned().unwrap_or_else(|| RatFn::int(0)) };
    let coeff_x = |q: &QPoly, m: usize| -> Rational {
        q.get(m).cloned().unwrap_or_else(|| Rational::from(0))
    };

    let mut matrix: Vec<Vec<Rational>> = Vec::new();
    let mut rhs: Vec<Rational> = Vec::new();
    for p in 0..max_p {
        // ℚ(x) equation: Σ cⱼₖ·col_rfⱼₖ = tgt_rf.
        let col_rf: Vec<RatFn> = n_cols.iter().map(|n| coeff_t(n, p)).collect();
        let tgt_rf = coeff_t(&n_target, p);

        // Clear the common x-denominator.
        let mut d_x = vec![Rational::from(1)];
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
            matrix.push(s_cols.iter().map(|s| coeff_x(s, m)).collect());
            rhs.push(coeff_x(&s_tgt, m));
        }
    }
    (matrix, rhs)
}

/// Solve `M·x = b` over ℚ by Gauss–Jordan elimination, returning a particular
/// solution (free variables set to 0) or `None` if inconsistent.
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
    // Consistency: an all-zero row with nonzero rhs ⇒ no solution.
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::risch::alg_field::{quotient_derivation, radical_dy};
    use crate::integrate::risch::number_field::Quotient;
    use rug::Rational;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }

    /// `x` as a `ℚ(x)(t)` constant-in-t element.
    fn x_elem() -> TExpr {
        TExpr::from_ratfn(RatFn::from_poly(&vec![rat(0), rat(1)]))
    }

    #[test]
    fn tower_derivation_of_t_is_eta_prime_t() {
        // η = x ⇒ η' = 1, t = eˣ, D(t) = t.
        let field = ExpTowerField::new(RatFn::int(1));
        let dt = field.derivation(&TExpr::t());
        assert_eq!(dt, TExpr::t());
        // D(x) = 1.
        assert_eq!(field.derivation(&x_elem()), TExpr::int(1));
        // D(x + t) = 1 + t.
        let x_plus_t = field.add(&x_elem(), &TExpr::t());
        let expected = field.add(&TExpr::int(1), &TExpr::t());
        assert_eq!(field.derivation(&x_plus_t), expected);
    }

    #[test]
    fn tower_quotient_rule() {
        // η = x.  D(1/t) = −η'·t / t² = −1/t.
        let field = ExpTowerField::new(RatFn::int(1));
        let inv_t = field.inv(&TExpr::t()).unwrap();
        let d = field.derivation(&inv_t);
        let neg_inv_t = field.neg(&inv_t);
        assert_eq!(d, neg_inv_t);
    }

    /// Build the radical extension `ℚ(x)(eˣ)[y]/(y³ − (x + eˣ))` and return
    /// `(quotient, dy)`.  This is the Example-15 field: a degree-3 radical whose
    /// radicand `x + eˣ` involves the transcendental.
    fn cbrt_x_plus_exp() -> (Quotient<ExpTowerField>, Vec<TExpr>) {
        let field = ExpTowerField::new(RatFn::int(1)); // t = eˣ
                                                       // modulus  y³ − (x + t):  [ −(x+t), 0, 0, 1 ].
        let neg_a = field.neg(&field.add(&x_elem(), &TExpr::t()));
        let modulus = vec![neg_a, field.zero(), field.zero(), field.one()];
        let q = Quotient::new(field, modulus);
        let dy = radical_dy(&q);
        (q, dy)
    }

    #[test]
    fn radical_over_exp_tower_derivation_consistency() {
        // In ℚ(x)(eˣ)[y]/(y³−(x+eˣ)):  D(y³) must equal D(x+eˣ) = 1 + eˣ.
        let (q, dy) = cbrt_x_plus_exp();
        let field = q.field().clone();

        // y³ reduces to the element (x + t).
        let one = field.one();
        let zero = field.zero();
        let y3 = q.reduce(&[zero.clone(), zero.clone(), zero, one]); // [0,0,0,1] mod q
        let x_plus_t = field.add(&x_elem(), &TExpr::t());
        assert!(
            q.elem_eq(&y3, std::slice::from_ref(&x_plus_t)),
            "y³ should reduce to x+eˣ"
        );

        // D(y³) via the extension derivation.
        let d_y3 = quotient_derivation(&q, &dy, &y3);
        let one_plus_t = field.add(&TExpr::int(1), &TExpr::t());
        assert!(
            q.elem_eq(&d_y3, &[one_plus_t]),
            "D(y³) = D(x+eˣ) = 1+eˣ; got {d_y3:?}"
        );
    }

    #[test]
    fn radical_over_exp_tower_leibniz() {
        // D(y·y²) = D(y)·y² + y·D(y²) in ℚ(x)(eˣ)(y).
        let (q, dy) = cbrt_x_plus_exp();
        let field = q.field().clone();
        let y = vec![field.zero(), field.one()]; // y
        let y2 = q.mul(&y, &y);

        let lhs = quotient_derivation(&q, &dy, &q.mul(&y, &y2));
        let rhs = q.add(
            &q.mul(&quotient_derivation(&q, &dy, &y), &y2),
            &q.mul(&y, &quotient_derivation(&q, &dy, &y2)),
        );
        assert!(
            q.elem_eq(&lhs, &rhs),
            "Leibniz over the tower: {lhs:?} vs {rhs:?}"
        );
    }

    #[test]
    fn solve_tower_rde_v_equals_t() {
        // v' + 0·v = t.  Since D(t) = t, v = t.
        let field = ExpTowerField::new(RatFn::int(1));
        let v = solve_tower_rde(&field, &field.zero(), &TExpr::t()).expect("v = t");
        assert!(field.eq(&v, &TExpr::t()), "got {v:?}");
    }

    #[test]
    fn solve_tower_rde_example15_component() {
        // The i=2 component of tutorial Example 15, in ℚ(x)(eˣ):
        //   ω₂ = (2/3)·(1+t)/(x+t),  c₂ = [(2x+3)t + 5x]/(x+t)  ⇒  v₂ = 3x.
        let field = ExpTowerField::new(RatFn::int(1)); // t = eˣ
        let a = field.add(&x_elem(), &TExpr::t()); // x + t
        let a_prime = field.derivation(&a); // 1 + t
        let two_thirds = TExpr::from_ratfn(RatFn::new(vec![rat(2)], vec![rat(3)]));
        let omega2 = field.mul(&two_thirds, &field.mul(&a_prime, &field.inv(&a).unwrap()));

        let num = vec![
            RatFn::from_poly(&vec![rat(0), rat(5)]), // 5x   · t⁰
            RatFn::from_poly(&vec![rat(3), rat(2)]), // 2x+3 · t¹
        ];
        let den = vec![
            RatFn::from_poly(&vec![rat(0), rat(1)]), // x · t⁰
            RatFn::int(1),                           // 1 · t¹
        ];
        let c2 = TExpr::new(num, den);

        let v = solve_tower_rde(&field, &omega2, &c2).expect("Example-15 component is solvable");
        let expected = TExpr::from_ratfn(RatFn::from_poly(&vec![rat(0), rat(3)])); // 3x
        assert!(field.eq(&v, &expected), "v₂ should be 3x; got {v:?}");
    }

    #[test]
    fn solve_tower_rde_with_denominator() {
        // v' = −(1+t)/(x+t)²  ⇒  v = 1/(x+t).  Needs a denominator — exercises
        // the lifted ansatz (D = 1 fails, a candidate denominator succeeds).
        let field = ExpTowerField::new(RatFn::int(1));
        let x_plus_t = field.add(&x_elem(), &TExpr::t());
        let one_plus_t = field.add(&TExpr::int(1), &TExpr::t());
        let xt_sq = field.mul(&x_plus_t, &x_plus_t);
        let c = field.neg(&field.mul(&one_plus_t, &field.inv(&xt_sq).unwrap()));

        let v = solve_tower_rde(&field, &field.zero(), &c).expect("solvable with a denominator");
        // The solver verifies internally; re-assert D(v) = c.
        assert!(
            field.eq(&field.derivation(&v), &c),
            "D(v) = c; got v = {v:?}"
        );
        // v·(x+t) should be 1 (modulo the homogeneous constant, which is 0 here).
        let prod = field.mul(&v, &x_plus_t);
        assert!(field.eq(&prod, &TExpr::int(1)), "v·(x+t) = 1; got {prod:?}");
    }

    #[test]
    fn solve_tower_rde_inv_t_now_solvable() {
        // v' = 1/t  ⇒  v = −1/t.  The lifted ansatz (denominators allowed) finds
        // it where the old polynomial-only solver returned None.
        let field = ExpTowerField::new(RatFn::int(1));
        let inv_t = field.inv(&TExpr::t()).unwrap();
        let v = solve_tower_rde(&field, &field.zero(), &inv_t).expect("v = −1/t");
        assert!(
            field.eq(&v, &field.neg(&inv_t)),
            "v should be −1/t; got {v:?}"
        );
    }

    #[test]
    fn solve_tower_rde_nonelementary_returns_none() {
        // v' = eˣ/x has no solution in ℚ(x)(eˣ) (∫eˣ/x = Ei, non-elementary):
        // the solver returns None for every candidate denominator.
        let field = ExpTowerField::new(RatFn::int(1));
        let c = field.mul(&TExpr::t(), &field.inv(&x_elem()).unwrap()); // t/x
        assert!(solve_tower_rde(&field, &field.zero(), &c).is_none());
    }

    #[test]
    fn radical_over_exp_tower_dy_value() {
        // D(y)·(3y²) = D(y³) = 1 + eˣ, i.e. 3y²·D(y) = 1+t.
        let (q, dy) = cbrt_x_plus_exp();
        let field = q.field().clone();
        let y = vec![field.zero(), field.one()];
        let y2 = q.mul(&y, &y);
        let three_y2 = q.mul(&[field.from_i64(3)], &y2);
        let prod = q.mul(&three_y2, &dy);
        let one_plus_t = field.add(&TExpr::int(1), &TExpr::t());
        assert!(
            q.elem_eq(&prod, &[one_plus_t]),
            "3y²·D(y) = 1+eˣ; got {prod:?}"
        );
    }
}
