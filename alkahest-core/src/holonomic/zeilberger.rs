//! Zeilberger's algorithm: creative telescoping for proper hypergeometric terms.
//!
//! Given a proper hypergeometric term `F(n, k)` (see [`super::hyperterm`]),
//! this searches for a **P-recursive** (holonomic) relation
//!
//! ```text
//! Σ_{i=0}^{J} a_i(n)·F(n+i, k) = G(n, k+1) − G(n, k),    G(n,k) = R(n,k)·F(n,k)
//! ```
//!
//! with polynomial coefficients `a_i(n)` (not all zero, `a_J ≢ 0`) and an
//! exact rational-function certificate `R(n,k)`. Summing both sides over `k`
//! telescopes the right-hand side, so if `S(n) = Σ_k F(n,k)` then
//! `Σ_i a_i(n)·S(n+i) = 0`: this is exactly the classical route from
//! `Σ_k C(n,k) = 2^n` (order 1) to `Σ_k C(n,k)^2 = C(2n,n)` (order 1, higher
//! degree certificate) and beyond.
//!
//! # Method
//!
//! This is the standard Gosper-style reduction (Petkovšek–Wilf–Zeilberger,
//! *A=B*, ch. 6; Koepf, *Hypergeometric Summation*, ch. 7), generalized from
//! `Q` to the field `Q(n)` using the [`super::qfield`] towers:
//!
//! 1. Write `p(n,k) = F(n,k+1)/F(n,k)` (a fixed element of `Q(n)(k)`,
//!    independent of the unknown `a_i`) and, for each `i`, `c_i(n,k) =
//!    F(n+i,k)/F(n,k)` — both computed exactly by [`super::hyperterm::ProperTerm`].
//! 2. Take `D(k)`, a common denominator of the `c_i` over `Q(n)[k]` (known and
//!    `a_i`-independent), and work with `W(n,k) = F(n,k)/D(k)`. Then
//!    `Σ_i a_i·F(n+i,k) = N(k)·W(n,k)` where `N(k) = Σ_i a_i·D(k)·c_i(k)` is a
//!    *polynomial*, linear in the unknowns. Decompose the shift ratio of `W`,
//!    `ρ(k) = p(k)·D(k)/D(k+1)`, into Gosper normal form
//!    `ρ = A(k)·C(k+1) / (B(k)·C(k))` over `Q(n)[k]` — the same shifted-gcd
//!    construction as `sum::gosper::gosper_normal_form` (private), lifted to
//!    the field `Q(n)`. (Normal-forming `p` itself instead of `ρ` loses the
//!    `D` bookkeeping and the equation below has no polynomial solution even
//!    for `F = C(n,k)`.)
//! 3. Gosper's key equation for the term `N(k)·W(k)` is then the *polynomial*
//!    identity `A(k)·X(k+1) − B(k−1)·X(k) = C(k)·N(k)`, linear in the unknowns
//!    `{a_i}` and the coefficients of `X`, with
//!    `G(n,k) = R(n,k)·F(n,k)`, `R = B(k−1)·X(k) / (C(k)·D(k))`.
//! 4. For increasing order `J = 1, 2, …` and certificate degree `d = 0, 1,
//!    …`, normalize the leading recurrence coefficient `a_J = 1` and solve
//!    the resulting linear system **over the field `Q(n)`** (Gaussian
//!    elimination with `Q(n)`-valued pivots) for the remaining `a_i` and the
//!    coefficients of `X`.
//! 5. [`super::qfield::clear_denominators`] turns the solved `a_i(n) ∈ Q(n)`
//!    into an integer-content-primitive polynomial family sharing one common
//!    scale `S(n)`; `R` is rescaled by the same `S(n)` (a `k`-independent
//!    factor, so this preserves the identity exactly).
//! 6. The *only* thing that makes a candidate a result: the rescaled
//!    `(a_i, R)` pair is plugged back in and checked as an exact `Q(n)(k)`
//!    identity (§ non-negotiable discipline — never return an unverified
//!    certificate). Only a verified candidate is returned; a verification
//!    failure discards the candidate and the search continues.

use super::hyperterm::{ratk_to_expr, ratuni_to_expr, ProperTerm};
use super::qfield::{
    clear_denominators, rn_div, rn_inv, rn_is_zero, rn_mul, rn_one, rn_poly, rn_sub, rn_zero,
    PolyK, RatK, Rn,
};
use super::HolonomicError;
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprId, ExprPool};
use crate::matrix::normal_form::RatUniPoly;

/// Search bounds for [`zeilberger()`].
#[derive(Debug, Clone, Copy)]
pub struct ZeilbergerOpts {
    /// Largest recurrence order `J` to try (searched from 1 upward).
    pub max_order: usize,
    /// Largest certificate-polynomial degree (in `k`) to try per order.
    pub max_degree: usize,
}

impl Default for ZeilbergerOpts {
    fn default() -> Self {
        ZeilbergerOpts {
            max_order: 4,
            max_degree: 16,
        }
    }
}

/// A verified Zeilberger certificate: `Σ_i coeffs[i](n)·F(n+i,k) = ΔG`,
/// `G(n,k) = certificate(n,k)·F(n,k)`.
#[derive(Debug, Clone)]
pub struct ZeilbergerResult {
    /// Recurrence order `J`; `coeffs.len() == order + 1`.
    pub order: usize,
    /// `a_0(n), …, a_J(n)` as expressions in `n`; integer-content-primitive,
    /// `a_J` not identically zero.
    pub coeffs: Vec<ExprId>,
    /// `R(n,k)` as an expression in `n, k`, with `G(n,k) = R(n,k)·F(n,k)`.
    pub certificate: ExprId,
}

/// `k^j` as an element of `Q(n)[k]`.
fn k_mono(j: usize) -> PolyK {
    let mut coeffs = vec![rn_zero(); j + 1];
    coeffs[j] = rn_one();
    PolyK::from_coeffs(coeffs)
}

/// Generalization of `sum::gosper::gosper_normal_form` from `Q` to
/// the field `Q(n)`: writes `p/q = Z·A(k)·C(k+1) / (B(k)·C(k))` with
/// `gcd(A(k), B(k+h))` a unit for every `h ≥ 0`, `Z` folded into `A`.
fn gosper_normal_form_qn(mut p: PolyK, mut q: PolyK) -> Option<(PolyK, PolyK, PolyK)> {
    if p.is_zero() {
        return Some((PolyK::zero(), PolyK::one(), PolyK::one()));
    }
    if q.is_zero() {
        return None;
    }
    let lc_p = p.leading_coeff();
    let lc_q = q.leading_coeff();
    let z_scale = rn_div(&lc_p, &lc_q)?;
    p = p.scale(&rn_inv(&lc_p)?);
    q = q.scale(&rn_inv(&lc_q)?);

    let mut a = p;
    let mut b = q;
    let mut c = PolyK::one();

    let bound = (a.degree().max(0) + b.degree().max(0)).max(1) as usize + 32;

    loop {
        let mut found = false;
        // `i = 0` matters: the normal form requires `gcd(A(k), B(k+h))` to be a
        // unit for every `h ≥ 0`, *including* a plain common factor at `h = 0`.
        // Starting at `i = 1` leaves such a factor in both `A` and `B` — which
        // silently breaks the ansatz, so the linear system at the true minimal
        // order has no solution and the search runs on to a larger order.
        for i in 0..=bound {
            let bshift = b.shift_k(i as i64);
            let d = PolyK::gcd(&a, &bshift);
            if d.is_zero() || d.degree() == 0 {
                continue;
            }
            let Some(an) = PolyK::exact_div(&a, &d) else {
                continue;
            };
            let dsmi = d.shift_k(-(i as i64));
            let Some(bn) = PolyK::exact_div(&b, &dsmi) else {
                continue;
            };
            a = an;
            b = bn;
            let mut prod = PolyK::one();
            for j in 1..=i {
                prod = prod.mul(&d.shift_k(-(j as i64)));
            }
            c = c.mul(&prod);
            found = true;
            break;
        }
        if !found {
            break;
        }
    }
    a = a.scale(&z_scale);
    Some((a, b, c))
}

/// Gaussian elimination over the field `Q(n)` (as opposed to `Q`): same
/// structure as `sum::gosper::rational_gaussian_solve`, generalized to `Rn`
/// pivots/division. A zero column is a free variable, set to zero. Returns
/// `None` when the system is inconsistent.
fn field_gaussian_solve(mut mat: Vec<Vec<Rn>>, mut rhs: Vec<Rn>) -> Option<Vec<Rn>> {
    let nrows = mat.len();
    if nrows == 0 {
        return Some(vec![]);
    }
    let ncols = mat[0].len();
    let mut row = 0;
    for col in 0..ncols {
        if row >= nrows {
            break;
        }
        let pr = (row..nrows).find(|&r| !rn_is_zero(&mat[r][col]));
        let Some(pr) = pr else {
            continue;
        };
        mat.swap(row, pr);
        rhs.swap(row, pr);
        let inv = rn_inv(&mat[row][col])?;
        for entry in mat[row].iter_mut().skip(col) {
            *entry = rn_mul(entry, &inv);
        }
        rhs[row] = rn_mul(&rhs[row], &inv);
        let pivot_row = mat[row].clone();
        let pivot_rhs = rhs[row].clone();
        for r in 0..nrows {
            if r == row {
                continue;
            }
            let v = mat[r][col].clone();
            if rn_is_zero(&v) {
                continue;
            }
            for (entry, pivot) in mat[r].iter_mut().zip(pivot_row.iter()).skip(col) {
                *entry = rn_sub(entry, &rn_mul(pivot, &v));
            }
            rhs[r] = rn_sub(&rhs[r], &rn_mul(&pivot_rhs, &v));
        }
        row += 1;
    }

    for (r, mrow) in mat.iter().enumerate() {
        let all_zero = mrow.iter().all(rn_is_zero);
        if all_zero && !rn_is_zero(&rhs[r]) {
            return None;
        }
    }

    let mut sol = vec![rn_zero(); ncols];
    for r in (0..nrows).rev() {
        let first = mat[r].iter().position(|e| !rn_is_zero(e));
        if let Some(j) = first {
            let mut sum = rhs[r].clone();
            for cidx in (j + 1)..ncols {
                sum = rn_sub(&sum, &rn_mul(&mat[r][cidx], &sol[cidx]));
            }
            sol[j] = rn_div(&sum, &mat[r][j])?;
        }
    }
    Some(sol)
}

/// Solve Gosper's key equation
///
/// ```text
/// A(k)·X(k+1) − B(k−1)·X(k) = C(k)·N(k),    N(k) = Σ_i a_i·C_i(k)
/// ```
///
/// for a degree-`d` polynomial `X` and the recurrence coefficients
/// `a_0..a_{order-1}` (with `a_order` normalized to `1`, so its term moves to
/// the right-hand side). `c_ci[i]` is `C(k)·C_i(k)`. Comparing coefficients of
/// each power of `k` gives a linear system over the field `Q(n)`; a solution
/// exists iff the candidate `(order, d)` pair admits a certificate.
/// Returns `(x_coeffs, lam_below_order)` on success.
fn try_solve(
    aa: &PolyK,
    b_eq: &PolyK,
    c_ci: &[PolyK],
    order: usize,
    d: usize,
) -> Option<(Vec<Rn>, Vec<Rn>)> {
    // BX_j(k) = A·(k+1)^j − B(k−1)·k^j, for j = 0..=d
    let mut bx: Vec<PolyK> = Vec::with_capacity(d + 1);
    for j in 0..=d {
        let kp1j = k_mono(j).shift_k(1);
        let kj = k_mono(j);
        let term_a = aa.mul(&kp1j);
        let term_b = b_eq.mul(&kj);
        bx.push(term_a.sub(&term_b));
    }

    // max degree across every basis polynomial (including the RHS mover, c_ci[order]).
    let mut max_deg = 0i32;
    for p in &bx {
        max_deg = max_deg.max(p.degree());
    }
    for p in c_ci {
        max_deg = max_deg.max(p.degree());
    }
    if max_deg < 0 {
        max_deg = 0;
    }
    let n_eq = (max_deg as usize) + 1;
    let n_var = (d + 1) + order; // x_0..x_d, lam_0..lam_{order-1}

    let mut mat = vec![vec![rn_zero(); n_var]; n_eq];
    let mut rhs = vec![rn_zero(); n_eq];

    for (m, row) in mat.iter_mut().enumerate() {
        for (j, bxj) in bx.iter().enumerate() {
            row[j] = bxj.coeff(m);
        }
        for i in 0..order {
            row[(d + 1) + i] = super::qfield::rn_neg(&c_ci[i].coeff(m));
        }
        rhs[m] = c_ci[order].coeff(m);
    }

    let sol = field_gaussian_solve(mat, rhs)?;
    let x_coeffs = sol[..=d].to_vec();
    let lam_below = sol[(d + 1)..].to_vec();
    Some((x_coeffs, lam_below))
}

/// Zeilberger's algorithm: find a verified P-recursive relation for a
/// proper hypergeometric term `F(n, k)` (see module docs).
///
/// Refuses with [`HolonomicError`] rather than guessing when `term` is not
/// a proper hypergeometric term in `(n, k)`, or when the bounded search in
/// `opts` finds no certificate that passes exact verification.
pub fn zeilberger(
    term: ExprId,
    n: ExprId,
    k: ExprId,
    pool: &ExprPool,
    opts: &ZeilbergerOpts,
) -> Result<DerivedExpr<ZeilbergerResult>, HolonomicError> {
    if n == k {
        return Err(HolonomicError::InvalidInput(
            "the outer index n and the summation index k must be distinct symbols".into(),
        ));
    }
    if opts.max_order == 0 || opts.max_degree == 0 {
        return Err(HolonomicError::InvalidInput(
            "max_order and max_degree must both be at least 1".into(),
        ));
    }

    let f = ProperTerm::parse(term, n, k, pool)?;
    let p = f.ratio_k()?;
    for order in 1..=opts.max_order {
        let c: Vec<RatK> = (0..=order as i64)
            .map(|i| f.ratio_n(i))
            .collect::<Result<_, _>>()?;

        // `D(k)`: a common denominator of the shift quotients `c_i`, over
        // `Q(n)[k]` and independent of the unknown `a_i`. Working with
        // `W(n,k) = F(n,k)/D(k)` is what keeps the Gosper equation polynomial:
        // `Σ_i a_i·F(n+i,k) = N(k)·W(n,k)` with `N = Σ_i a_i·C_i` polynomial
        // and linear in the unknowns.
        let mut dden = PolyK::one();
        for ci in &c {
            dden = PolyK::lcm(&dden, &ci.den);
        }
        if dden.is_zero() {
            continue;
        }
        // `C_i(k) = D(k)·c_i(k) ∈ Q(n)[k]` — polynomial by construction of `D`.
        let ci_polys: Option<Vec<PolyK>> = c
            .iter()
            .map(|ci| PolyK::exact_div(&dden.mul(&ci.num), &ci.den))
            .collect();
        let Some(ci_polys) = ci_polys else {
            continue;
        };

        // Gosper normal form of the shift ratio of `W`, *not* of `p`:
        // `ρ(k) = W(k+1)/W(k) = p(k)·D(k)/D(k+1) = A(k)·C(k+1) / (B(k)·C(k))`.
        // Normal-forming `p` alone would drop the `D` bookkeeping and the
        // resulting equation would have no polynomial solution even for
        // `F = C(n,k)`.
        let rho_num = p.num.mul(&dden);
        let rho_den = p.den.mul(&dden.shift_k(1));
        let Some((aa, bb, cc)) = gosper_normal_form_qn(rho_num, rho_den) else {
            continue;
        };
        let b_eq = bb.shift_k(-1);

        // Gosper's key equation for the term `N(k)·W(k)`:
        //     A(k)·X(k+1) − B(k−1)·X(k) = C(k)·N(k),
        // so the right-hand basis the unknown `a_i` multiply is `C(k)·C_i(k)`.
        let c_ci: Vec<PolyK> = ci_polys.iter().map(|q| cc.mul(q)).collect();

        for d in 0..=opts.max_degree {
            let Some((x_coeffs, lam_below)) = try_solve(&aa, &b_eq, &c_ci, order, d) else {
                continue;
            };

            let mut lam_full = lam_below;
            lam_full.push(rn_one()); // a_order = 1

            let x_poly = PolyK::from_coeffs(x_coeffs);
            // `G(k) = (B(k−1)·X(k) / C(k))·W(k) = R(k)·F(n,k)` with
            // `R(k) = B(k−1)·X(k) / (C(k)·D(k))`.
            let r_pre = RatK {
                num: b_eq.mul(&x_poly),
                den: cc.mul(&dden),
            }
            .normalize();

            let a_int: Vec<RatUniPoly> = clear_denominators(&lam_full);
            if a_int.iter().all(|p| p.is_zero()) {
                continue;
            }
            let scale = rn_poly(a_int[order].clone());
            if rn_is_zero(&scale) {
                continue;
            }
            let r_final = RatK {
                num: r_pre.num.scale(&scale),
                den: r_pre.den.clone(),
            }
            .normalize();

            // Exact verification: Σ_i a_i(n)·c_i(n,k) ?= R(n,k+1)·p(n,k) − R(n,k).
            let mut lhs = RatK::zero();
            for (i, ci) in c.iter().enumerate() {
                let ai = RatK::from_rn(rn_poly(a_int[i].clone()));
                lhs = lhs.add(&ai.mul(ci));
            }
            let rhs_check = r_final.shift_k(1).mul(&p).sub(&r_final);
            if !lhs.sub(&rhs_check).is_zero() {
                // Refuse silently and keep searching: never return an
                // unverified certificate.
                continue;
            }

            let coeffs_expr: Vec<ExprId> =
                a_int.iter().map(|p| ratuni_to_expr(pool, n, p)).collect();
            let certificate_expr = ratk_to_expr(pool, n, k, &r_final);

            let mut log = DerivationLog::new();
            log.push(RewriteStep::simple(
                "zeilberger_certificate",
                term,
                certificate_expr,
            ));

            return Ok(DerivedExpr::with_log(
                ZeilbergerResult {
                    order,
                    coeffs: coeffs_expr,
                    certificate: certificate_expr,
                },
                log,
            ));
        }
    }

    Err(HolonomicError::SearchExhausted(format!(
        "no verified P-recursive relation of order <= {} with certificate degree <= {} \
         in k was found for {}",
        opts.max_order,
        opts.max_degree,
        pool.display(term)
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn nk(pool: &ExprPool) -> (ExprId, ExprId) {
        (
            pool.symbol("n", Domain::Real),
            pool.symbol("k", Domain::Real),
        )
    }

    fn binom(pool: &ExprPool, top: ExprId, bot: ExprId) -> ExprId {
        let g1 = pool.func("gamma", vec![pool.add(vec![top, pool.integer(1_i32)])]);
        let g2 = pool.func("gamma", vec![pool.add(vec![bot, pool.integer(1_i32)])]);
        let g3 = pool.func(
            "gamma",
            vec![pool.add(vec![
                top,
                pool.mul(vec![bot, pool.integer(-1_i32)]),
                pool.integer(1_i32),
            ])],
        );
        pool.mul(vec![
            g1,
            pool.pow(g2, pool.integer(-1_i32)),
            pool.pow(g3, pool.integer(-1_i32)),
        ])
    }

    /// Σ_k C(n,k) = 2^n : Zeilberger should find the order-1 recurrence
    /// a_1(n)·F(n+1,k) + a_0(n)·F(n,k) = ΔG with a_1 = 1, a_0 = -2 (up to a
    /// common integer scale).
    #[test]
    fn binomial_row_sum_order_one() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let f = binom(&pool, n, k);
        let opts = ZeilbergerOpts::default();
        let result = zeilberger(f, n, k, &pool, &opts).expect("Zeilberger must find a certificate");
        let r = &result.value;
        assert_eq!(r.order, 1, "expected order-1 recurrence for Σ_k C(n,k)");
        assert_eq!(r.coeffs.len(), 2);
        // The recurrence must be S(n+1) − 2·S(n) = 0 up to an overall scale:
        // a_0(n) + 2·a_1(n) ≡ 0. Checked numerically at several n, which is
        // enough to pin a ratio of low-degree polynomials.
        for ni in [3.0_f64, 7.0, 11.5] {
            let env = std::collections::HashMap::from([(n, ni)]);
            let a0 = crate::eval_f64(r.coeffs[0], &pool, &env).expect("a_0(n) evaluates");
            let a1 = crate::eval_f64(r.coeffs[1], &pool, &env).expect("a_1(n) evaluates");
            assert!(a1.abs() > 1e-12, "leading coefficient must not vanish");
            assert!(
                (a0 / a1 + 2.0).abs() < 1e-9,
                "expected a_0/a_1 = -2 (S(n+1) = 2·S(n)), got {}",
                a0 / a1
            );
        }
    }

    /// Refuses non-hypergeometric input rather than guessing.
    #[test]
    fn refuses_non_hypergeometric_input() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let bad = pool.func("sin", vec![pool.mul(vec![n, k])]);
        let opts = ZeilbergerOpts::default();
        let err = zeilberger(bad, n, k, &pool, &opts).expect_err("sin(nk) is not hypergeometric");
        assert!(matches!(err, HolonomicError::NotProperHypergeometric(_)));
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-HOLO-001");
    }

    /// n == k is refused as invalid input, not silently misinterpreted.
    #[test]
    fn refuses_coincident_indices() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let opts = ZeilbergerOpts::default();
        let err = zeilberger(n, n, n, &pool, &opts).expect_err("n == k must be refused");
        assert!(matches!(err, HolonomicError::InvalidInput(_)));
    }

    /// Every returned certificate — not just the happy-path example above —
    /// satisfies the exact Q(n)(k) identity it claims to.
    #[test]
    fn certificate_reverifies_exactly() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let f = binom(&pool, n, k);
        let opts = ZeilbergerOpts::default();
        let result = zeilberger(f, n, k, &pool, &opts).expect("certificate");
        let r = &result.value;

        let term = ProperTerm::parse(f, n, k, &pool).expect("parse");
        let c: Vec<RatK> = (0..=r.order as i64)
            .map(|i| term.ratio_n(i).expect("ratio_n"))
            .collect();
        let p = term.ratio_k().expect("ratio_k");

        // Re-derive the algebraic coefficients from the returned expressions
        // via the same parser used for hypergeometric prefactors, and check
        // the identity independently of the internal search state.
        let a: Vec<RatK> = r
            .coeffs
            .iter()
            .map(|&e| {
                let ratk = super::super::hyperterm::as_ratk(e, n, k, &pool, 0)
                    .expect("coeff must be a function of n alone");
                assert_eq!(ratk.num.degree().max(0), 0, "coeff must not depend on k");
                assert_eq!(ratk.den.degree(), 0, "coeff must not depend on k");
                ratk
            })
            .collect();
        let r_ratk = super::super::hyperterm::as_ratk(r.certificate, n, k, &pool, 0)
            .expect("certificate must parse back into Q(n)(k)");

        let mut lhs = RatK::zero();
        for (i, ci) in c.iter().enumerate() {
            lhs = lhs.add(&a[i].mul(ci));
        }
        let rhs = r_ratk.shift_k(1).mul(&p).sub(&r_ratk);
        assert!(
            lhs.sub(&rhs).is_zero(),
            "returned certificate must satisfy the exact identity"
        );
    }
}
