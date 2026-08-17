//! The Apagodu–Zeilberger ansatz search: given a proper hypergeometric
//! `F(n,j,k)`, find `a_0(n), …, a_J(n)` (not all zero) and two rational
//! certificates `c_1, c_2 ∈ Q(n,j,k)` such that
//!
//! ```text
//! Σ_i a_i(n)·F(n+i,j,k) = Δ_j G_1 + Δ_k G_2,   G_1 = c_1·F,  G_2 = c_2·F
//! ```
//!
//! # Method — undetermined coefficients, not a 2-D Gosper normal form
//!
//! The single-sum engine ([`super::super::zeilberger`]) puts the shift ratio
//! of `F` into *Gosper normal form* before solving, which is what lets it
//! search a smaller ansatz efficiently. There is no standard two-dimensional
//! analogue of that normal form for a general proper hypergeometric `F(n,j,k)`
//! — this is exactly why Apagodu–Zeilberger's method (unlike single-index
//! Zeilberger) is usually presented as an undetermined-coefficients search:
//! posit a certificate of bounded degree over a *fixed* denominator, clear
//! it, and solve the resulting linear system.
//!
//! Concretely: divide the target identity by `F(n,j,k)`. Writing
//! `ρ_j(n,j,k) = F(n,j+1,k)/F(n,j,k) = N_j/D_j` and
//! `ρ_k(n,j,k) = F(n,j,k+1)/F(n,j,k) = N_k/D_k` (both known rational
//! functions, computed exactly by [`super::term::ProperTerm3`]), and taking
//! the certificate ansatz `c_1 = P_1(n,j,k)/D_j(n,j,k)`,
//! `c_2 = P_2(n,j,k)/D_k(n,j,k)` with `P_1, P_2` polynomials of bounded
//! degree, the identity becomes, after multiplying through by the (known,
//! ansatz-independent) common denominator
//!
//! ```text
//! D_total = (∏_i D_{n,i}) · D_j(n,j,k)·D_j(n,j+1,k) · D_k(n,j,k)·D_k(n,j,k+1)
//! ```
//!
//! a **polynomial** identity in `Q[n,j,k]`, linear in the unknown
//! coefficients of `a_i(n) = Σ_p α_{i,p}·n^p` and of `P_1, P_2`. Matching
//! coefficients of every monomial `n^s·j^q·k^r` gives one linear equation per
//! monomial; [`solve_ansatz`] assembles that system and takes its nullspace
//! over `Q` by plain Gaussian elimination (see [`rational_nullspace`]).
//!
//! `D_j(n,j,k)` (the *raw*, un-reduced denominator of `ρ_j`) is not the
//! minimal possible certificate denominator in general — a genuine 2-D Gosper
//! reduction would sometimes need a smaller one after cancelling a
//! shift-equivalent factor between `N_j` and a shifted `D_j` (exactly what
//! the single-sum engine's `C(k)` factor exists to supply). This module does
//! **not** compute that reduction. For the ordinary "binomial-type" double
//! sums this targets, the shift ratios are already close to normal form
//! (`gcd(N_j(j), D_j(j+h))` is a unit for every `h ≥ 0` in the examples this
//! module is tested against), so the raw denominator is already sufficient —
//! but this is a property of the *examples*, not a theorem the code
//! establishes. When it is not sufficient, the bounded search below simply
//! finds nothing and reports [`Telescoping2dError::SearchExhausted`]; it
//! never claims a false certificate, because every candidate is re-verified
//! from scratch (see [`verify_certificate`]) before it is returned.

use super::poly::{Axis, Poly3, Rat3};
use super::term::ProperTerm3;
use super::Telescoping2dError;
use crate::kernel::{ExprId, ExprPool};
use rug::{Integer, Rational};
use std::collections::BTreeMap;

/// Search bounds for [`telescope2d`](super::telescope2d). All three are
/// genuine upper bounds — the search tries every combination
/// `1..=max_order × 0..=max_a_degree × 0..=max_cert_degree` in ascending
/// order (cheapest first in each axis), so raising them only admits harder
/// inputs.
///
/// Unlike [`super::super::zeilberger::ZeilbergerOpts`] this is **not**
/// cost-ordered across the three axes jointly (see the module docs): the
/// three loops are simply nested, ascending. That is a real scope
/// simplification, not an oversight — the point of the cost-ordered plan in
/// the single-sum engine is to make expensive high-degree probes at low
/// order not block a cheap high-order solution; the double-sum ansatz here
/// does not yet have that tuning.
#[derive(Debug, Clone, Copy)]
pub struct Telescoping2dOpts {
    /// Largest recurrence order `J`; orders are tried from 1 upward.
    pub max_order: usize,
    /// Largest degree (in `n`) tried for each `a_i(n)`.
    pub max_a_degree: usize,
    /// Largest *box* degree (in each of `n`, `j`, `k` independently) tried
    /// for the certificate numerators `P_1(n,j,k)`, `P_2(n,j,k)`.
    pub max_cert_degree: usize,
}

impl Default for Telescoping2dOpts {
    fn default() -> Self {
        Telescoping2dOpts {
            max_order: 2,
            max_a_degree: 2,
            max_cert_degree: 3,
        }
    }
}

/// A verified double-sum creative-telescoping certificate.
///
/// The verified content is the identity
/// `Σ_i a_i(n)·F(n+i,j,k) = Δ_j G_1 + Δ_k G_2` with `G_1 = cert1·F`,
/// `G_2 = cert2·F` — checked exactly in `Q(n,j,k)` by `verify_certificate`
/// before this is ever constructed. Turning it into a recurrence for
/// `S(n) = Σ_j Σ_k F(n,j,k)` over a stated range is a separate step; see
/// [`super::boundary`].
#[derive(Debug, Clone)]
pub struct Telescoping2dResult {
    pub order: usize,
    /// `a_0(n), …, a_J(n)`, integer-content-primitive as a family (the same
    /// discipline as the single-sum engine's `clear_denominators`, applied to
    /// the flat vector of every unknown in this certificate — see
    /// `primitive_scale_rationals`).
    pub coeffs: Vec<ExprId>,
    /// `c_1(n,j,k)`, with `G_1 = c_1·F`.
    pub cert1: ExprId,
    /// `c_2(n,j,k)`, with `G_2 = c_2·F`.
    pub cert2: ExprId,
}

/// Internal (pre-`ExprId`) form of a candidate, kept in algebraic form so
/// [`verify_certificate`] can re-check it without any expression-pool
/// round-trip.
struct Candidate {
    order: usize,
    a: Vec<Poly3>, // a_i(n), i = 0..=order, degree only in the N axis
    c1: Rat3,
    c2: Rat3,
}

/// Apagodu–Zeilberger search: find and verify a double-sum certificate for
/// `term`, a proper hypergeometric `F(n,j,k)`.
pub fn telescope2d_search(
    term: ExprId,
    n: ExprId,
    j: ExprId,
    k: ExprId,
    pool: &ExprPool,
    opts: &Telescoping2dOpts,
) -> Result<Telescoping2dResult, Telescoping2dError> {
    if n == j || n == k || j == k {
        return Err(Telescoping2dError::InvalidInput(
            "n, j and k must be three distinct symbols".into(),
        ));
    }
    if opts.max_order == 0 {
        return Err(Telescoping2dError::InvalidInput(
            "max_order must be at least 1".into(),
        ));
    }

    let f = ProperTerm3::parse(term, n, j, k, pool)?;
    let rho_j = f.ratio_axis(Axis::J, 1)?;
    let rho_k = f.ratio_axis(Axis::K, 1)?;
    if rho_j.den.is_zero() || rho_k.den.is_zero() {
        return Err(Telescoping2dError::NotProperHypergeometric(
            "shift ratio has a zero denominator".into(),
        ));
    }

    for order in 1..=opts.max_order {
        // The `i = 0..=order` shift ratios in `n` do not depend on the
        // degree budgets, so they are computed once per order.
        let mut nn = Vec::with_capacity(order + 1);
        let mut dn = Vec::with_capacity(order + 1);
        for i in 0..=order as i64 {
            let r = f.ratio_axis(Axis::N, i)?;
            if r.den.is_zero() {
                return Err(Telescoping2dError::NotProperHypergeometric(
                    "shift ratio in n has a zero denominator".into(),
                ));
            }
            nn.push(r.num);
            dn.push(r.den);
        }

        for a_degree in 0..=opts.max_a_degree {
            for cert_degree in 0..=opts.max_cert_degree {
                if let Some(cand) =
                    solve_ansatz(order, a_degree, cert_degree, &nn, &dn, &rho_j, &rho_k)?
                {
                    if verify_certificate(&f, &cand) {
                        return Ok(finish(cand, n, j, k, pool));
                    }
                    // A genuine implementation bug, not a user-facing error:
                    // never happens for a correct construction, but refusing
                    // silently and continuing keeps the "never return an
                    // unverified certificate" discipline absolute.
                }
            }
        }
    }

    Err(Telescoping2dError::SearchExhausted(format!(
        "no verified double-sum certificate of order <= {} was found for {} \
         within a_degree <= {} and certificate degree <= {}",
        opts.max_order,
        pool.display(term),
        opts.max_a_degree,
        opts.max_cert_degree
    )))
}

/// Build and solve the linear system for one `(order, a_degree, cert_degree)`
/// probe. `Ok(None)` means the system has no solution with a non-trivial
/// leading coefficient — not an error, just this probe failing.
#[allow(clippy::too_many_arguments)]
// `p`, `q`, `r` below double as monomial exponents (via `idx_p1`/`idx_p2`)
// and as indices into the memoized `(j+1)^q`/`(k+1)^r` power tables, so
// clippy's "use an iterator instead" rewrite does not apply cleanly.
#[allow(clippy::needless_range_loop)]
fn solve_ansatz(
    order: usize,
    a_degree: usize,
    cert_degree: usize,
    nn: &[Poly3],
    dn: &[Poly3],
    rho_j: &Rat3,
    rho_k: &Rat3,
) -> Result<Option<Candidate>, Telescoping2dError> {
    let dj = &rho_j.den;
    let nj = &rho_j.num;
    let dk = &rho_k.den;
    let nk = &rho_k.num;

    // The certificate denominators are *not* just the raw `D_j`/`D_k` of the
    // single ratio being telescoped: a certificate built from a product of
    // two single-sum WZ pairs (see the worked "separable" example in
    // `mod.rs`) needs a factor from the *other* direction's `n`-shift ratio
    // too — e.g. `c_1 = R_A(n,j)·[B(n+1,k)/B(n,k)]` for `F = A(n,j)·B(n,k)`,
    // whose denominator is not a function of `j` alone. So the ansatz here
    // is `c_1 = P_1 / E_1`, `E_1 := D_j · (∏_i D_{n,i})`, and symmetrically
    // for `c_2` with `D_k`. This is still just a *fixed, ansatz-independent*
    // denominator (no gcd, no minimality claim) — see the module docs — just
    // a bigger one than the raw single-ratio denominator alone, chosen to be
    // sufficient for the worked examples this module is tested against.
    let mut dnfull = Poly3::one();
    for d in dn {
        dnfull = dnfull.mul(d);
    }
    let mut dnfull_excl: Vec<Poly3> = Vec::with_capacity(order + 1);
    for i in 0..=order {
        let mut p = Poly3::one();
        for (idx, d) in dn.iter().enumerate() {
            if idx != i {
                p = p.mul(d);
            }
        }
        dnfull_excl.push(p);
    }

    let e1 = dj.mul(&dnfull);
    let e1s = e1.shift(Axis::J, 1);
    let e2 = dk.mul(&dnfull);
    let e2s = e2.shift(Axis::K, 1);

    // `Mn_i = D_total / D_{n,i}`, built by replacing `E_1`'s copy of
    // `D_{n,i}` with the `i`-excluded product (any one of `D_total`'s four
    // redundant copies of `D_{n,i}` may be the one "removed" — they are all
    // equal, so the result is the same polynomial either way).
    let mut mn: Vec<Poly3> = Vec::with_capacity(order + 1);
    for excl in &dnfull_excl {
        let e1_excl_i = dj.mul(excl);
        mn.push(e1_excl_i.mul(&e1s).mul(dj).mul(&e2).mul(&e2s).mul(dk));
    }
    let mj = e2.mul(&e2s).mul(dk);
    let mk = e1.mul(&e1s).mul(dj);

    // Unknown layout: a_{i,p} for i=0..=order, p=0..=a_degree; then P1
    // coefficients over the box n<=cert_degree, j<=cert_degree, k<=cert_degree;
    // then P2 coefficients over the same box.
    let na = a_degree + 1;
    let box_len = cert_degree + 1;
    let cert_count = box_len * box_len * box_len;
    let a_count = (order + 1) * na;
    let total = a_count + 2 * cert_count;

    let idx_a = |i: usize, p: usize| i * na + p;
    let idx_p1 = |p: usize, q: usize, r: usize| a_count + (p * box_len + q) * box_len + r;
    let idx_p2 =
        |p: usize, q: usize, r: usize| a_count + cert_count + (p * box_len + q) * box_len + r;

    // Monomial -> row (dense-in-practice sparse map of coefficient vectors).
    let mut rows: BTreeMap<(u32, u32, u32), Vec<Rational>> = BTreeMap::new();
    let mut add_contribution = |mono: &BTreeMap<(u32, u32, u32), Rational>, col: usize| {
        for (exp, c) in mono {
            let row = rows
                .entry(*exp)
                .or_insert_with(|| vec![Rational::from(0); total]);
            row[col] += c.clone();
        }
    };

    for i in 0..=order {
        for p in 0..=a_degree {
            let basis = Poly3::var(Axis::N)
                .pow_u32(p as u32)
                .mul(&nn[i])
                .mul(&mn[i]);
            add_contribution(&basis.terms, idx_a(i, p));
        }
    }
    // Memoized (j+1)^q and (k+1)^r.
    let mut jp1_pow: Vec<Poly3> = Vec::with_capacity(box_len);
    let mut kp1_pow: Vec<Poly3> = Vec::with_capacity(box_len);
    {
        let jp1 = Poly3::var(Axis::J).add(&Poly3::one());
        let kp1 = Poly3::var(Axis::K).add(&Poly3::one());
        let mut acc_j = Poly3::one();
        let mut acc_k = Poly3::one();
        for _ in 0..box_len {
            jp1_pow.push(acc_j.clone());
            kp1_pow.push(acc_k.clone());
            acc_j = acc_j.mul(&jp1);
            acc_k = acc_k.mul(&kp1);
        }
    }
    for p in 0..=cert_degree {
        let np = Poly3::var(Axis::N).pow_u32(p as u32);
        for q in 0..=cert_degree {
            for r in 0..=cert_degree {
                let kr = Poly3::var(Axis::K).pow_u32(r as u32);
                // P1(n,j+1,k) term: n^p (j+1)^q k^r ; P1(n,j,k) term: n^p j^q k^r.
                // P1(n,j+1,k)*Nj*E1 - P1(n,j,k)*E1s*Dj, times Mj — see the
                // derivation in the module docs: this is
                // `[c1(j+1)*rho_j - c1(j)] * (E1*E1s*Dj) * Mj`.
                let mono_shift = np.mul(&jp1_pow[q]).mul(&kr);
                let mono_plain = np.mul(&Poly3::var(Axis::J).pow_u32(q as u32)).mul(&kr);
                let basis1 = mono_shift
                    .mul(nj)
                    .mul(&e1)
                    .sub(&mono_plain.mul(&e1s).mul(dj))
                    .mul(&mj)
                    .neg();
                add_contribution(&basis1.terms, idx_p1(p, q, r));

                let jq = Poly3::var(Axis::J).pow_u32(q as u32);
                let mono_shift_k = np.mul(&jq).mul(&kp1_pow[r]);
                let mono_plain_k = np.mul(&jq).mul(&kr);
                let basis2 = mono_shift_k
                    .mul(nk)
                    .mul(&e2)
                    .sub(&mono_plain_k.mul(&e2s).mul(dk))
                    .mul(&mk)
                    .neg();
                add_contribution(&basis2.terms, idx_p2(p, q, r));
            }
        }
    }

    let matrix: Vec<Vec<Rational>> = rows.into_values().collect();
    let basis = rational_nullspace(matrix, total);
    if basis.is_empty() {
        return Ok(None);
    }

    for vec in &basis {
        // Reject a basis vector whose top a_order(n) is identically zero:
        // that is really a lower-order (or no) relation, and returning it
        // under this `order` would misreport the recurrence's order.
        let top_nonzero = (0..na).any(|p| vec[idx_a(order, p)] != 0);
        if !top_nonzero {
            continue;
        }
        let mut scaled = vec.clone();
        primitive_scale_rationals(&mut scaled);

        let mut a = Vec::with_capacity(order + 1);
        for i in 0..=order {
            let mut p = Poly3::zero();
            for pw in 0..=a_degree {
                let c = scaled[idx_a(i, pw)].clone();
                if c != 0 {
                    p = p.add(&Poly3::var(Axis::N).pow_u32(pw as u32).scale(&c));
                }
            }
            a.push(p);
        }
        let mut p1 = Poly3::zero();
        let mut p2 = Poly3::zero();
        for p in 0..=cert_degree {
            for q in 0..=cert_degree {
                for r in 0..=cert_degree {
                    let c1c = scaled[idx_p1(p, q, r)].clone();
                    if c1c != 0 {
                        p1 = p1.add(&monomial3(p as u32, q as u32, r as u32, c1c));
                    }
                    let c2c = scaled[idx_p2(p, q, r)].clone();
                    if c2c != 0 {
                        p2 = p2.add(&monomial3(p as u32, q as u32, r as u32, c2c));
                    }
                }
            }
        }
        let c1 = Rat3 {
            num: p1,
            den: e1.clone(),
        };
        let c2 = Rat3 {
            num: p2,
            den: e2.clone(),
        };
        return Ok(Some(Candidate { order, a, c1, c2 }));
    }
    Ok(None)
}

fn monomial3(en: u32, ej: u32, ek: u32, c: Rational) -> Poly3 {
    Poly3::var(Axis::N)
        .pow_u32(en)
        .mul(&Poly3::var(Axis::J).pow_u32(ej))
        .mul(&Poly3::var(Axis::K).pow_u32(ek))
        .scale(&c)
}

/// Scale a flat family of rationals to integer, content-1 (up to sign) —
/// the same principle as [`super::super::qfield::make_primitive`], applied
/// to one combined vector spanning `a_i(n)` *and* both certificate
/// numerators, because a homogeneous linear relation stays a solution under
/// any single overall rescaling. Kept local rather than reusing
/// `make_primitive` because that helper is specialized to a family of
/// single-variable `RatUniPoly`s; here the family spans three different
/// polynomial rings (one univariate, two trivariate) sharing only their
/// scalar coefficients.
fn primitive_scale_rationals(v: &mut [Rational]) {
    let mut den_lcm = Integer::from(1);
    for c in v.iter() {
        den_lcm = den_lcm.lcm(c.denom());
    }
    for c in v.iter_mut() {
        *c *= den_lcm.clone();
    }
    let mut content = Integer::from(0);
    for c in v.iter() {
        content = content.gcd(c.numer());
    }
    if content != 0 && content != 1 {
        let inv = Rational::from((1, content.clone()));
        for c in v.iter_mut() {
            *c *= inv.clone();
        }
    }
    let sign_ref = v.iter().rev().find(|c| **c != 0).cloned();
    if let Some(s) = sign_ref {
        if s < 0 {
            for c in v.iter_mut() {
                *c *= Rational::from(-1);
            }
        }
    }
}

/// Nullspace basis of `mat` (rows = equations, `ncols` unknowns) over `Q`, by
/// plain Gaussian elimination to row-echelon form.
// The pivot column `col` bounds the range read/written below; a plain slice
// iterator would need to skip a variable, non-slice-aligned prefix on both
// `mat[row]` and every other row simultaneously, so the index form is
// clearer here than clippy's suggested rewrite.
#[allow(clippy::needless_range_loop)]
fn rational_nullspace(mut mat: Vec<Vec<Rational>>, ncols: usize) -> Vec<Vec<Rational>> {
    let nrows = mat.len();
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut row = 0;
    for col in 0..ncols {
        if row >= nrows {
            break;
        }
        let Some(pr) = (row..nrows).find(|&r| mat[r][col] != 0) else {
            continue;
        };
        mat.swap(row, pr);
        let inv = Rational::from(1) / mat[row][col].clone();
        for c in col..ncols {
            mat[row][c] *= inv.clone();
        }
        for r in 0..nrows {
            if r == row {
                continue;
            }
            let factor = mat[r][col].clone();
            if factor == 0 {
                continue;
            }
            for c in col..ncols {
                let sub = factor.clone() * mat[row][c].clone();
                mat[r][c] -= sub;
            }
        }
        pivot_cols.push(col);
        row += 1;
    }
    let is_pivot: Vec<bool> = {
        let mut v = vec![false; ncols];
        for &c in &pivot_cols {
            v[c] = true;
        }
        v
    };
    let mut basis = Vec::new();
    for free_col in 0..ncols {
        if is_pivot[free_col] {
            continue;
        }
        let mut v = vec![Rational::from(0); ncols];
        v[free_col] = Rational::from(1);
        for (r, &pc) in pivot_cols.iter().enumerate() {
            v[pc] = -mat[r][free_col].clone();
        }
        basis.push(v);
    }
    basis
}

/// Re-derive and check the telescoping identity **from scratch** — never
/// trusting the linear-algebra search that produced `cand`. This mirrors
/// `zeilberger()`'s own final check and `PositivityCertificate::verify`'s
/// discipline: a candidate only becomes a result after this passes.
fn verify_certificate(f: &ProperTerm3, cand: &Candidate) -> bool {
    let mut lhs = Rat3::zero();
    for (i, ai) in cand.a.iter().enumerate() {
        let Ok(ratio) = f.ratio_axis(Axis::N, i as i64) else {
            return false;
        };
        lhs = lhs.add(&Rat3::from_poly(ai.clone()).mul(&ratio));
    }
    let Ok(rho_j) = f.ratio_axis(Axis::J, 1) else {
        return false;
    };
    let Ok(rho_k) = f.ratio_axis(Axis::K, 1) else {
        return false;
    };
    let g1_delta = cand.c1.shift(Axis::J, 1).mul(&rho_j).sub(&cand.c1);
    let g2_delta = cand.c2.shift(Axis::K, 1).mul(&rho_k).sub(&cand.c2);
    let rhs = g1_delta.add(&g2_delta);
    lhs.eq_rat(&rhs)
}

fn finish(
    cand: Candidate,
    n: ExprId,
    j: ExprId,
    k: ExprId,
    pool: &ExprPool,
) -> Telescoping2dResult {
    let coeffs = cand.a.iter().map(|p| p.to_expr(pool, n, j, k)).collect();
    Telescoping2dResult {
        order: cand.order,
        coeffs,
        cert1: cand.c1.to_expr(pool, n, j, k),
        cert2: cand.c2.to_expr(pool, n, j, k),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn njk(pool: &ExprPool) -> (ExprId, ExprId, ExprId) {
        (
            pool.symbol("n", Domain::Real),
            pool.symbol("j", Domain::Real),
            pool.symbol("k", Domain::Real),
        )
    }

    fn binom(pool: &ExprPool, top: ExprId, bot: ExprId) -> ExprId {
        pool.func("binomial", vec![top, bot])
    }

    /// `F(n,j,k) = C(n,j)*C(j,k)`, `Σ_j Σ_k F = Σ_j C(n,j)*2^j`. Not itself
    /// the closed-form check (that lives in `mod.rs`'s worked-example test);
    /// this is the narrower claim that the ansatz search finds *some*
    /// verified telescoping certificate for a genuinely non-separable proper
    /// hypergeometric term in three indices.
    #[test]
    fn finds_a_certificate_for_double_binomial_product() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let f = pool.mul(vec![binom(&pool, n, j), binom(&pool, j, k)]);
        let opts = Telescoping2dOpts::default();
        let result = telescope2d_search(f, n, j, k, &pool, &opts)
            .expect("a double-sum certificate should be found for C(n,j)*C(j,k)");
        assert!(result.order >= 1);
        assert_eq!(result.coeffs.len(), result.order + 1);
    }

    #[test]
    fn refuses_coincident_indices() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let opts = Telescoping2dOpts::default();
        let err =
            telescope2d_search(n, n, n, n, &pool, &opts).expect_err("n, j, k must be distinct");
        assert!(matches!(err, Telescoping2dError::InvalidInput(_)));
    }
}
