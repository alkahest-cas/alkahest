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
//! exact rational-function certificate `R(n,k)`. That identity — and only that
//! identity — is what [`zeilberger()`] verifies exactly before returning.
//!
//! # The sum recurrence carries a hypothesis
//!
//! Summing both sides over `k = κ₀ .. κ₁` telescopes the right-hand side to a
//! **boundary difference**, not to zero:
//!
//! ```text
//! Σ_i a_i(n)·S(n+i) = G(n, κ₁+1) − G(n, κ₀),    S(n) = Σ_{k=κ₀}^{κ₁} F(n,k)
//! ```
//!
//! `Σ_i a_i(n)·S(n+i) = 0` therefore holds **only when that boundary difference
//! vanishes** — the *natural boundary* hypothesis, which is what makes the
//! classical route from `Σ_k C(n,k) = 2ⁿ` to `Σ_k C(n,k)² = C(2n,n)` work: `G`
//! is a rational multiple of `F`, and `F` vanishes outside `0 ≤ k ≤ n`.
//!
//! It is not automatic. For `F(n,k) = C(n,k)/(k+1)` summed over `k = 0..n` the
//! certificate is correct and `G(n,0) = −1`, so
//! `(n+2)·S(n+1) − (2n+2)·S(n) = 1`, not `0`; `S(n) = (2ⁿ⁺¹−1)/(n+1)` confirms
//! it in exact arithmetic. A caller who reads the homogeneous recurrence off a
//! certificate without checking the boundary gets a false lemma.
//!
//! **[`super::boundary::boundary_status`] decides it** over a stated summation
//! range, three-valued: proved to vanish, proved nonzero (with the
//! inhomogeneity `b(n)` explicit), or undecided — in which case nothing about
//! the sum may be claimed. [`boundary_term`] still returns `G(n,k)` for a caller
//! who would rather discharge the hypothesis by hand, and
//! [`boundary_side_condition`] states it in words.
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
//! 4. For a candidate order `J` and certificate degree `d`, normalize the
//!    leading recurrence coefficient `a_J = 1` and solve the resulting linear
//!    system **over the field `Q(n)`** (Gaussian elimination with
//!    `Q(n)`-valued pivots) for the remaining `a_i` and the coefficients of
//!    `X`. The `(J, d)` pairs are visited by **iterative deepening**, cheapest
//!    estimated cost first (see the private `search_plan`), so `max_order` and
//!    `max_degree` are genuine upper bounds: raising them admits harder inputs
//!    without slowing down inputs that are decided early. Cheapest-first is
//!    *not* order-ascending, so the order it returns is not minimal unless the
//!    search happened to establish it — see [`OrderSearch`] and
//!    [`ZeilbergerSearchReport::order_is_minimal`], and use
//!    [`OrderSearch::MinimalOrder`] when minimality is the claim being made.
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
///
/// Both fields are upper *bounds*, not starting points: the search deepens
/// through the `(order, degree)` grid cheapest-first (see the private
/// `search_plan`), so
/// raising either one only widens what can be found — it does not make an input
/// that was already decided any slower.
///
/// Under [`OrderSearch::MinimalOrder`] that stops being true of `max_degree`,
/// necessarily: it is the bound minimality is claimed *against*, so the whole
/// sweep up to it has to happen at every order below the answer.
#[derive(Debug, Clone, Copy)]
pub struct ZeilbergerOpts {
    /// Largest recurrence order `J` to try; orders are searched from 1 upward.
    pub max_order: usize,
    /// Largest certificate-polynomial degree (in `k`) to try, per order.
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
///
/// The verified content is the **telescoping identity in `k`**. Turning it into
/// a recurrence for `S(n) = Σ_k F(n,k)` needs the boundary difference
/// `G(n, κ₁+1) − G(n, κ₀)` to vanish over the summation range; see the module
/// documentation, [`boundary_term`] and [`boundary_side_condition`].
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

/// `G(n,k) = R(n,k)·F(n,k)`, the telescoped quantity whose boundary values
/// decide whether the certificate's recurrence holds for the *sum*.
///
/// `term` must be the same `F(n,k)` that was passed to [`zeilberger()`]. The
/// recurrence for `S(n) = Σ_{k=κ₀}^{κ₁} F(n,k)` is
/// `Σ_i a_i(n)·S(n+i) = G(n, κ₁+1) − G(n, κ₀)`, so this is exactly what a
/// caller needs in order to discharge (or refute) the natural-boundary
/// hypothesis for their own summation range.
pub fn boundary_term(result: &ZeilbergerResult, term: ExprId, pool: &ExprPool) -> ExprId {
    crate::simplify::simplify(pool.mul(vec![result.certificate, term]), pool).value
}

/// The hypothesis that [`ZeilbergerResult`]'s recurrence for the *sum* rests on,
/// stated in words, for a caller who wants to record it without deciding it.
///
/// It is a fixed string, which is precisely why it is no longer what the Python
/// binding reports: an invariant caveat reads identically for a case where the
/// hypothesis holds and one where it fails, so nothing that only reads it can
/// tell the two apart. [`super::boundary::boundary_status`] computes a verdict
/// instead, and [`super::boundary::BoundaryStatus::side_conditions`] is what the
/// binding emits.
pub const fn boundary_side_condition() -> &'static str {
    "the recurrence Σ_i a_i(n)·S(n+i) = 0 for S(n) = Σ_k F(n,k) additionally requires \
     G(n, k_hi+1) = G(n, k_lo) over the summation range, where G(n,k) = R(n,k)·F(n,k); \
     Zeilberger verifies the telescoping identity in k, not this boundary condition. \
     It holds for the usual natural boundary (F vanishing outside 0 <= k <= n) and fails \
     for e.g. F = C(n,k)/(k+1), where G(n,0) = -1 makes the recurrence inhomogeneous"
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

/// Everything the search needs for one recurrence order that does *not* depend
/// on the certificate degree `d`.
///
/// Built once per order and reused by every degree the iterative deepening
/// later revisits that order at: the shift quotients, the common denominator
/// `D(k)` and the Gosper normal form are all `d`-independent, and recomputing
/// them per degree would make the interleaved search pay for itself many times
/// over.
struct OrderState {
    /// `c_i(n,k) = F(n+i,k)/F(n,k)` for `i = 0..=order`.
    c: Vec<RatK>,
    /// `D(k)`, the common denominator of the `c_i`.
    dden: PolyK,
    /// `A(k)` from the Gosper normal form of `ρ(k) = p(k)·D(k)/D(k+1)`.
    aa: PolyK,
    /// `B(k−1)`, i.e. the `B` of the normal form shifted for the key equation.
    b_eq: PolyK,
    /// `C(k)` from the Gosper normal form.
    cc: PolyK,
    /// `C(k)·C_i(k)` — the basis the unknown `a_i` multiply in the key equation.
    c_ci: Vec<PolyK>,
}

/// Degree-independent setup for one recurrence order.
///
/// `Ok(None)` means this order is structurally unusable (a degenerate common
/// denominator or a shift ratio with no Gosper normal form) and the search
/// should skip it, as opposed to `Err`, which aborts because the input is not a
/// proper hypergeometric term at all.
fn order_state(
    f: &ProperTerm,
    p: &RatK,
    order: usize,
) -> Result<Option<OrderState>, HolonomicError> {
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
        return Ok(None);
    }
    // `C_i(k) = D(k)·c_i(k) ∈ Q(n)[k]` — polynomial by construction of `D`.
    let ci_polys: Option<Vec<PolyK>> = c
        .iter()
        .map(|ci| PolyK::exact_div(&dden.mul(&ci.num), &ci.den))
        .collect();
    let Some(ci_polys) = ci_polys else {
        return Ok(None);
    };

    // Gosper normal form of the shift ratio of `W`, *not* of `p`:
    // `ρ(k) = W(k+1)/W(k) = p(k)·D(k)/D(k+1) = A(k)·C(k+1) / (B(k)·C(k))`.
    // Normal-forming `p` alone would drop the `D` bookkeeping and the
    // resulting equation would have no polynomial solution even for
    // `F = C(n,k)`.
    let rho_num = p.num.mul(&dden);
    let rho_den = p.den.mul(&dden.shift_k(1));
    let Some((aa, bb, cc)) = gosper_normal_form_qn(rho_num, rho_den) else {
        return Ok(None);
    };
    let b_eq = bb.shift_k(-1);

    // Gosper's key equation for the term `N(k)·W(k)`:
    //     A(k)·X(k+1) − B(k−1)·X(k) = C(k)·N(k),
    // so the right-hand basis the unknown `a_i` multiply is `C(k)·C_i(k)`.
    let c_ci: Vec<PolyK> = ci_polys.iter().map(|q| cc.mul(q)).collect();

    Ok(Some(OrderState {
        c,
        dden,
        aa,
        b_eq,
        cc,
        c_ci,
    }))
}

/// How many certificate-degree steps one extra recurrence order is worth when
/// the search decides what to try next.
///
/// Measured, not guessed: on `Σ (−1)^k C(n,k)³` one `try_solve` probe costs
/// ≈ 3× more per unit of `d` (0.7 ms at `d = 0`, 0.6 s at `d = 7`, 84 s at
/// `d = 12`) and ≈ 30× more per unit of `order` at fixed `d` — and `30 ≈ 3³`.
/// So `order + 1` costs about what `d + 3` costs, and sweeping the frontier
/// `3·(order−1) + d = t` keeps every probe in one pass at comparable cost.
const ORDER_COST_IN_DEGREE_STEPS: usize = 3;

/// How [`zeilberger_search`] walks the `(order, degree)` grid.
///
/// The two modes answer different questions, and the difference is not a
/// tuning knob: [`OrderSearch::CostOrdered`] answers *is there a verified
/// relation within these bounds*, [`OrderSearch::MinimalOrder`] answers *what
/// is the least order of any relation within these bounds*. Only the second
/// one can report [`ZeilbergerSearchReport::order_is_minimal`] as `true` in
/// general, and only the first one is fast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrderSearch {
    /// Cheapest-estimated-cost-first, the default and what
    /// [`zeilberger()`] uses.
    ///
    /// A returned order `J > 1` does **not** establish that no order-`J−1`
    /// relation exists: the plan may reach a cheap high-order probe before an
    /// expensive low-order one, which is exactly what makes Dixon, Franel and
    /// Apéry decidable at the default bounds at all.
    #[default]
    CostOrdered,
    /// Order-ascending: every degree `0..=max_degree` at order `J` is probed
    /// and rejected before order `J+1` is tried at all.
    ///
    /// A returned order is therefore genuinely the least one reachable within
    /// `max_degree`. The price is the whole hopeless low-order sweep the
    /// cost-ordered plan exists to avoid — on Franel and Apéry that is the
    /// difference between sub-second and tens of seconds (see the module
    /// tests). Ask for it when minimality is the result you want to publish,
    /// not as a default.
    MinimalOrder,
}

/// A [`ZeilbergerResult`] together with what the search that produced it does
/// and does not establish.
///
/// Returned by [`zeilberger_search`]; [`zeilberger()`] drops it to the bare
/// result for callers that do not care.
#[derive(Debug, Clone)]
pub struct ZeilbergerSearchReport {
    /// The verified certificate.
    pub result: ZeilbergerResult,
    /// `true` **only** when the search established that no relation of lower
    /// order exists at any degree `0..=max_degree`.
    ///
    /// `false` means *not established*, never *a lower order exists* — a
    /// lower-order relation that had been found would have been returned
    /// instead. Under [`OrderSearch::CostOrdered`] this is `true` for order 1
    /// (nothing is lower) and whenever the cost-ordered plan happened to
    /// exhaust every lower order first, and `false` otherwise; under
    /// [`OrderSearch::MinimalOrder`] it is always `true`.
    ///
    /// It is computed from the probes actually made, not from the mode, so it
    /// cannot drift away from what the search did.
    pub order_is_minimal: bool,
    /// How many `(order, degree)` probes the search made, the successful one
    /// included. This is the cost measure the two [`OrderSearch`] modes differ
    /// in; the tests use it to pin the difference.
    pub probes: usize,
}

/// The `(order, degree)` pairs to probe, in the order `search` asks for.
///
/// Every pair in `1..=max_order × 0..=max_degree` appears exactly once under
/// either mode, so an exhausted search does the same work it always did; only
/// the *order of visits* changes, which is what makes the bounds bounds rather
/// than starting points.
///
/// [`OrderSearch::CostOrdered`] sweeps the frontier `3·(order−1) + d = t`,
/// cheapest estimated cost first, ties going to the lower recurrence order.
/// The one deliberate trade-off: a term whose order-1 certificate needs a much
/// higher degree than its order-2 one (`d₁ > d₂ + 3`) gets the order-2
/// relation, where an order-major sweep would have insisted on order 1. Both
/// are verified relations; the preference for the sharper one cost minutes, or
/// never, and made the whole search unusable in practice.
///
/// [`OrderSearch::MinimalOrder`] is that order-major sweep, restored as an
/// opt-in for callers who need the sharper answer and will pay for it.
fn search_plan(max_order: usize, max_degree: usize, search: OrderSearch) -> Vec<(usize, usize)> {
    let mut plan = Vec::with_capacity(max_order * (max_degree + 1));
    match search {
        OrderSearch::MinimalOrder => {
            for order in 1..=max_order {
                for d in 0..=max_degree {
                    plan.push((order, d));
                }
            }
        }
        OrderSearch::CostOrdered => {
            let max_budget = ORDER_COST_IN_DEGREE_STEPS * (max_order - 1) + max_degree;
            for budget in 0..=max_budget {
                for order in 1..=max_order {
                    let spent = ORDER_COST_IN_DEGREE_STEPS * (order - 1);
                    if let Some(d) = budget.checked_sub(spent) {
                        if d <= max_degree {
                            plan.push((order, d));
                        }
                    }
                }
            }
        }
    }
    plan
}

/// Zeilberger's algorithm: find a verified P-recursive relation for a
/// proper hypergeometric term `F(n, k)` (see module docs).
///
/// `opts.max_order` and `opts.max_degree` are *bounds*, not starting points:
/// the search deepens through them (see below), so raising them permits harder
/// inputs without making easy ones more expensive.
///
/// Refuses with [`HolonomicError`] rather than guessing when `term` is not
/// a proper hypergeometric term in `(n, k)`, or when the bounded search in
/// `opts` finds no certificate that passes exact verification.
///
/// **The returned order is not claimed to be minimal.** The cost-ordered plan
/// can reach a cheap high-order probe before an expensive low-order one, so an
/// order-2 result does not rule out an order-1 relation. Use
/// [`zeilberger_search`] with [`OrderSearch::MinimalOrder`] when that matters;
/// its report says so explicitly rather than leaving it to be assumed.
pub fn zeilberger(
    term: ExprId,
    n: ExprId,
    k: ExprId,
    pool: &ExprPool,
    opts: &ZeilbergerOpts,
) -> Result<DerivedExpr<ZeilbergerResult>, HolonomicError> {
    zeilberger_search(term, n, k, pool, opts, OrderSearch::CostOrdered)
        .map(|d| d.map(|report| report.result))
}

/// [`zeilberger()`] with the grid traversal chosen by the caller, reporting
/// whether the order it returns is known to be minimal.
///
/// `search` decides the traversal only. What counts as a result — an exactly
/// re-verified `Q(n)(k)` identity — and the bounds in `opts` are the same
/// either way, and so is the boundary hypothesis a *sum* recurrence rests on
/// (see the module docs).
pub fn zeilberger_search(
    term: ExprId,
    n: ExprId,
    k: ExprId,
    pool: &ExprPool,
    opts: &ZeilbergerOpts,
    search: OrderSearch,
) -> Result<DerivedExpr<ZeilbergerSearchReport>, HolonomicError> {
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

    // Iterative deepening over the `(order, degree)` grid, cheapest probe first
    // (see [`search_plan`]), instead of the degree sweep nested inside the order
    // loop this used to be.
    //
    // The old nesting ran the entire `d = 0..=max_degree` sweep at order 1 before
    // it ever tried order 2, and a probe's cost grows roughly like `3^d` (Dixon,
    // order 1: 0.7 ms at `d = 0`, 0.6 s at `d = 7`, 84 s at `d = 12` — the `Q(n)`
    // Gaussian elimination widens *and* its rational-function entries grow with
    // `d`). So for the order ≥ 2 identities — Dixon, Franel, Apéry — the tail of
    // a hopeless order-1 sweep swamped everything, and `max_degree` acted as a
    // starting point: raising it made easy inputs catastrophically slower rather
    // than merely admitting harder ones. Deepening makes both `max_order` and
    // `max_degree` genuine upper bounds that cost nothing on inputs decided early.
    //
    // Nothing is skipped: the plan still visits every `(order, degree)` pair, so
    // an exhausted search does exactly the work it did before, and a relation
    // reachable within the old bounds is still reachable within the new ones.
    //
    // `OrderSearch::MinimalOrder` opts out of exactly that reordering, walking
    // the grid order-major so a returned order is the least one reachable
    // within `max_degree` — paying the hopeless low-order sweep to buy the
    // claim. Which mode ran is not what `order_is_minimal` is read off, though:
    // `degrees_failed` below counts the probes that actually happened, so the
    // flag stays true to the search even if the plan changes again.
    let mut states: Vec<Option<OrderState>> = Vec::with_capacity(opts.max_order);
    let mut degrees_failed = vec![0usize; opts.max_order];
    for (order, d) in search_plan(opts.max_order, opts.max_degree, search) {
        // Counted before the probe rather than after, because every path out of
        // this body other than `return` is a failure at `(order, d)` — and the
        // minimality test below only ever reads *lower* orders, which are
        // finished by then. `probes` is the same tally summed over the orders,
        // so the two can never disagree about what the search did.
        degrees_failed[order - 1] += 1;
        // `OrderState` is degree-independent, so each order is set up once no
        // matter how often the deepening returns to it.
        while states.len() < order {
            states.push(order_state(&f, &p, states.len() + 1)?);
        }
        let Some(state) = &states[order - 1] else {
            continue;
        };

        let Some((x_coeffs, lam_below)) = try_solve(&state.aa, &state.b_eq, &state.c_ci, order, d)
        else {
            continue;
        };

        let mut lam_full = lam_below;
        lam_full.push(rn_one()); // a_order = 1

        let x_poly = PolyK::from_coeffs(x_coeffs);
        // `G(k) = (B(k−1)·X(k) / C(k))·W(k) = R(k)·F(n,k)` with
        // `R(k) = B(k−1)·X(k) / (C(k)·D(k))`.
        let r_pre = RatK {
            num: state.b_eq.mul(&x_poly),
            den: state.cc.mul(&state.dden),
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
        for (i, ci) in state.c.iter().enumerate() {
            let ai = RatK::from_rn(rn_poly(a_int[i].clone()));
            lhs = lhs.add(&ai.mul(ci));
        }
        let rhs_check = r_final.shift_k(1).mul(&p).sub(&r_final);
        if !lhs.sub(&rhs_check).is_zero() {
            // Refuse silently and keep searching: never return an
            // unverified certificate.
            continue;
        }

        let coeffs_expr: Vec<ExprId> = a_int.iter().map(|p| ratuni_to_expr(pool, n, p)).collect();
        let certificate_expr = ratk_to_expr(pool, n, k, &r_final);

        let mut log = DerivationLog::new();
        log.push(RewriteStep::simple(
            "zeilberger_certificate",
            term,
            certificate_expr,
        ));

        // Minimal exactly when every strictly lower order was probed at every
        // degree in bounds and none of them yielded a verified certificate.
        // Vacuously true at order 1. Under `MinimalOrder` the plan guarantees
        // it; under `CostOrdered` it is usually false for order > 1, which is
        // the honest answer and the reason this flag exists.
        let order_is_minimal = (1..order).all(|j| degrees_failed[j - 1] == opts.max_degree + 1);

        return Ok(DerivedExpr::with_log(
            ZeilbergerSearchReport {
                result: ZeilbergerResult {
                    order,
                    coeffs: coeffs_expr,
                    certificate: certificate_expr,
                },
                order_is_minimal,
                probes: degrees_failed.iter().sum(),
            },
            log,
        ));
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

    /// `F(n,k) = C(n,k)/(k+1)` is the counterexample the boundary hypothesis
    /// exists for: the telescoping certificate is correct, but `G(n,0) = −1`, so
    /// `Σ_i a_i(n)·S(n+i)` is `1`, not `0`. The boundary term must therefore be
    /// available to the caller, and it must not vanish here.
    #[test]
    fn boundary_term_is_available_and_nonzero_where_the_hypothesis_fails() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let kp1 = pool.add(vec![k, pool.integer(1_i32)]);
        let f = pool.mul(vec![
            binom(&pool, n, k),
            pool.pow(kp1, pool.integer(-1_i32)),
        ]);
        let opts = ZeilbergerOpts::default();
        let result = zeilberger(f, n, k, &pool, &opts).expect("certificate");
        let g = boundary_term(&result.value, f, &pool);

        // G(n, 0) = R(n,0)·F(n,0) = −1 for every n, so the homogeneous sum
        // recurrence is false and nothing in the certificate said otherwise.
        let mut m = std::collections::HashMap::new();
        m.insert(k, pool.integer(0_i32));
        let g_at_0 = crate::simplify::simplify(crate::kernel::subs(g, &m, &pool), &pool).value;
        for ni in [2.0_f64, 5.0, 9.0] {
            let env = std::collections::HashMap::from([(n, ni)]);
            let v = crate::eval_f64(g_at_0, &pool, &env).expect("G(n,0) evaluates");
            assert!(
                (v + 1.0).abs() < 1e-9,
                "G({ni}, 0) should be -1, got {v} — the boundary difference does not vanish"
            );
        }
        assert!(boundary_side_condition().contains("G(n, k_hi+1) = G(n, k_lo)"));
    }

    /// `Σ_k C(n,k)³` and `Σ_k (−1)^k C(n,k)³` summed against the recurrence the
    /// certificate claims — the end-to-end check that the coefficients describe
    /// the sum they were derived for, independent of the certificate machinery.
    fn assert_annihilates(coeffs: &[f64], s: impl Fn(u64) -> f64, ni: u64, what: &str) {
        let total: f64 = coeffs
            .iter()
            .enumerate()
            .map(|(i, ai)| ai * s(ni + i as u64))
            .sum();
        let scale = coeffs
            .iter()
            .enumerate()
            .map(|(i, ai)| (ai * s(ni + i as u64)).abs())
            .fold(1.0_f64, f64::max);
        assert!(
            total.abs() < 1e-6 * scale,
            "{what}: recurrence must annihilate S(n) at n = {ni}, got {total}"
        );
    }

    fn coeffs_at(r: &ZeilbergerResult, n: ExprId, pool: &ExprPool, ni: u64) -> Vec<f64> {
        let env = std::collections::HashMap::from([(n, ni as f64)]);
        r.coeffs
            .iter()
            .map(|&e| crate::eval_f64(e, pool, &env).expect("a_i(n) evaluates"))
            .collect()
    }

    /// `C(n,k)` as an `f64`.
    fn binom_f64(m: u64, j: u64) -> f64 {
        (1..=j).fold(1.0_f64, |acc, t| acc * (m - t + 1) as f64 / t as f64)
    }

    /// Dixon: `Σ_k (−1)^k C(n,k)³` needs order 2, and it must be decided **at the
    /// default bounds**.
    ///
    /// Regression test for the search order. With the degree loop nested inside
    /// the order loop this ran the whole `d = 0..=max_degree` sweep at order 1 —
    /// where no relation exists at any degree, and where the last few degrees cost
    /// minutes each — before it ever tried order 2. Every order ≥ 2 identity was
    /// therefore unreachable at the shipped defaults while being seconds away at
    /// `max_degree = 4`, i.e. `max_degree` behaved as a starting point rather than
    /// a bound. If that regresses, this test does not fail subtly: it hangs.
    #[test]
    fn dixon_order_two_is_reachable_at_default_bounds() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let c = binom(&pool, n, k);
        let sign = pool.pow(pool.integer(-1_i32), k);
        let f = pool.mul(vec![sign, c, c, c]);

        let start = std::time::Instant::now();
        let result = zeilberger(f, n, k, &pool, &ZeilbergerOpts::default())
            .expect("Dixon must be decided at the default bounds");
        println!(
            "dixon: order {} in {:?}",
            result.value.order,
            start.elapsed()
        );

        let r = &result.value;
        assert_eq!(
            r.order, 2,
            "Σ_k (−1)^k C(n,k)³ satisfies an order-2 relation"
        );
        assert_eq!(r.coeffs.len(), 3);

        // S(n) = Σ_k (−1)^k C(n,k)³ — the natural boundary applies (F vanishes
        // outside 0 ≤ k ≤ n), so the homogeneous recurrence must hold.
        let s = |m: u64| -> f64 {
            (0..=m)
                .map(|j| {
                    let b = binom_f64(m, j);
                    if j % 2 == 0 {
                        b * b * b
                    } else {
                        -(b * b * b)
                    }
                })
                .sum()
        };
        for ni in 2_u64..8 {
            assert_annihilates(&coeffs_at(r, n, &pool, ni), s, ni, "dixon");
        }
    }

    /// Franel: `Σ_k C(n,k)³`, the same regression as Dixon but on the heavier of
    /// the two terms — and the wall-clock regression for the post-search
    /// normalisation.
    ///
    /// This used to be `#[ignore]`d at ~30 s, of which the search was 0.22 s and
    /// everything else was `RatK::normalize` running a Euclidean remainder
    /// sequence over `Q(n)` (`r_pre`, `r_final`, and the re-verification, ~10 s
    /// each). With the gcd done in `Z[n][k]` instead it is well under a second,
    /// so the test runs by default and asserts a bound that only the old
    /// coefficient blowup can breach. The bound is deliberately loose: the
    /// failure mode it guards against is two orders of magnitude, not a factor
    /// of two. Verification is unchanged — the certificate is still checked as
    /// an exact `Q(n)(k)` identity before it is returned.
    #[test]
    fn franel_order_two_is_reachable_at_default_bounds() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let c = binom(&pool, n, k);
        let f = pool.mul(vec![c, c, c]);

        let start = std::time::Instant::now();
        let result = zeilberger(f, n, k, &pool, &ZeilbergerOpts::default())
            .expect("Franel must be decided at the default bounds");
        let elapsed = start.elapsed();
        println!("franel: order {} in {:?}", result.value.order, elapsed);
        // The wall-clock guard is meaningful only in an uninstrumented release
        // build. Under a sanitizer it is not: the nightly `lsan` shard runs a
        // debug build with LeakSanitizer, where the whole lib suite takes ~23
        // minutes and this test breached a 10 s bound while the thing it guards
        // — the Z[n][k] gcd — was perfectly healthy. A timing assertion that
        // fails for the instrumentation rather than the regression is a flaky
        // test, so it is skipped there; the correctness assertions below always
        // run, in every configuration.
        // `debug_assertions` is the discriminator: every sanitizer shard builds
        // in debug, and the release run this guard is written for does not.
        if !cfg!(debug_assertions) {
            assert!(
                elapsed < std::time::Duration::from_secs(10),
                "Franel took {elapsed:?} at the default bounds — the exact Q(n)(k) \
                 post-processing has regressed to the coefficient blowup it used to have"
            );
        }

        let r = &result.value;
        assert_eq!(r.order, 2, "Σ_k C(n,k)³ satisfies an order-2 recurrence");
        assert_eq!(r.coeffs.len(), 3);

        let s = |m: u64| -> f64 {
            (0..=m)
                .map(|j| {
                    let b = binom_f64(m, j);
                    b * b * b
                })
                .sum()
        };
        for ni in 2_u64..7 {
            assert_annihilates(&coeffs_at(r, n, &pool, ni), s, ni, "franel");
        }
    }

    /// The deepening plan is a *reordering* of the grid, not a subset of it: it
    /// must visit every `(order, degree)` pair inside the caller's bounds exactly
    /// once (no gap could hide a relation, no repeat could waste a probe) and
    /// never step outside them.
    ///
    /// Both traversals are held to it: `MinimalOrder` buys its sharper claim by
    /// *reordering* the same grid, not by probing more of it, so an exhausted
    /// search costs the same either way and only a decided one differs.
    #[test]
    fn search_plan_is_a_permutation_of_the_bounded_grid() {
        for search in [OrderSearch::CostOrdered, OrderSearch::MinimalOrder] {
            for max_order in 1..6usize {
                for max_degree in 1..20usize {
                    assert_plan_is_a_permutation(max_order, max_degree, search);
                }
            }
        }
    }

    fn assert_plan_is_a_permutation(max_order: usize, max_degree: usize, search: OrderSearch) {
        let plan = search_plan(max_order, max_degree, search);
        let mut sorted = plan.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            plan.len(),
            "no pair may be probed twice ({max_order}, {max_degree}, {search:?})"
        );
        assert_eq!(
            plan.len(),
            max_order * (max_degree + 1),
            "every pair in the bounds must be probed ({max_order}, {max_degree}, {search:?})"
        );
        assert!(
            plan.iter()
                .all(|&(o, d)| (1..=max_order).contains(&o) && d <= max_degree),
            "the plan must stay inside the caller's bounds ({search:?})"
        );
    }

    /// Cheap probes come first, and a tie goes to the lower order — that is the
    /// whole content of the deepening, so it is asserted rather than implied.
    #[test]
    fn search_plan_visits_cheap_probes_first() {
        let plan = search_plan(3, 8, OrderSearch::CostOrdered);
        let cost = |&(o, d): &(usize, usize)| ORDER_COST_IN_DEGREE_STEPS * (o - 1) + d;
        assert!(
            plan.windows(2).all(|w| cost(&w[0]) <= cost(&w[1])),
            "probes must be visited in nondecreasing estimated cost: {plan:?}"
        );
        assert_eq!(plan[0], (1, 0), "the cheapest probe is order 1, degree 0");
        // The first four probes stay at order 1: an order-1 relation of low
        // degree is still found before anything of higher order, as before.
        assert_eq!(&plan[..4], &[(1, 0), (1, 1), (1, 2), (1, 3)]);
        // …and order 2 enters exactly when it becomes competitive.
        assert_eq!(plan[4], (2, 0));
    }

    /// The reason `MinimalOrder` can claim minimality at all: it does not reach
    /// order `J+1` until every degree in bounds at order `J` has been refused.
    ///
    /// This is the property `order_is_minimal` is derived from, asserted on the
    /// plan directly so a regression shows up here rather than as a flag that
    /// quietly starts lying.
    #[test]
    fn minimal_order_plan_exhausts_each_order_before_the_next() {
        let plan = search_plan(3, 8, OrderSearch::MinimalOrder);
        assert!(
            plan.windows(2)
                .all(|w| w[0].0 < w[1].0 || (w[0].0 == w[1].0 && w[0].1 < w[1].1)),
            "order-major, degree-ascending within an order: {plan:?}"
        );
        assert_eq!(&plan[..3], &[(1, 0), (1, 1), (1, 2)]);
        // Order 2 starts only once all nine degrees at order 1 are spent.
        assert_eq!(plan[9], (2, 0));
    }

    /// Order 1 needs no search to be minimal, and the default plan says so.
    ///
    /// Σ_k C(n,k) = 2ⁿ is decided at `(1, 0)`, the first probe under either
    /// traversal; nothing is lower than order 1, so the flag is `true` without
    /// the cost-ordered plan having to establish anything.
    #[test]
    fn order_one_is_minimal_under_the_default_plan() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let f = binom(&pool, n, k);
        let report = zeilberger_search(
            f,
            n,
            k,
            &pool,
            &ZeilbergerOpts::default(),
            OrderSearch::CostOrdered,
        )
        .expect("certificate");
        assert_eq!(report.value.result.order, 1);
        assert!(
            report.value.order_is_minimal,
            "order 1 is minimal by definition"
        );
        assert_eq!(report.value.probes, 1, "decided by the very first probe");
    }

    /// The honesty requirement, stated as a test: on Dixon the default plan
    /// returns order 2 **without** claiming it is minimal, because it reaches
    /// `(2, 0)` after only four of the seventeen order-1 probes in bounds.
    ///
    /// The premise that an ascending search makes minimality free is false for
    /// this plan, and this is where that would be caught. `MinimalOrder` on the
    /// same input does establish it — at a cost the test measures rather than
    /// assumes.
    #[test]
    fn default_plan_does_not_claim_minimality_it_has_not_established() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let c = binom(&pool, n, k);
        let sign = pool.pow(pool.integer(-1_i32), k);
        let f = pool.mul(vec![sign, c, c, c]);

        let opts = ZeilbergerOpts::default();
        let fast = zeilberger_search(f, n, k, &pool, &opts, OrderSearch::CostOrdered)
            .expect("Dixon is decided at the default bounds");
        assert_eq!(fast.value.result.order, 2);
        assert!(
            !fast.value.order_is_minimal,
            "the cost-ordered plan skipped order-1 degrees 4..=16, so it cannot \
             claim order 2 is minimal — and must not"
        );
        assert!(
            fast.value.probes <= opts.max_degree,
            "the whole point of the cost-ordered plan is that it decides Dixon \
             long before the order-1 sweep is done, got {} probes",
            fast.value.probes
        );
    }

    /// `MinimalOrder` at bounds it can afford: Dixon really has no order-1
    /// relation of certificate degree ≤ 4, so the order-2 answer is minimal
    /// there and the flag says so.
    ///
    /// `max_degree` is 4 rather than the default 16 because the order-1 sweep is
    /// what minimality costs and it grows like `3^d` — the point of the test is
    /// the flag, not the wall clock. A caller who needs minimality against a
    /// wider degree bound pays that bound; see the release notes for the
    /// measured Franel/Apéry numbers.
    #[test]
    fn minimal_mode_establishes_minimality() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let c = binom(&pool, n, k);
        let sign = pool.pow(pool.integer(-1_i32), k);
        let f = pool.mul(vec![sign, c, c, c]);

        let opts = ZeilbergerOpts {
            max_order: 4,
            max_degree: 4,
        };
        let start = std::time::Instant::now();
        let sharp = zeilberger_search(f, n, k, &pool, &opts, OrderSearch::MinimalOrder)
            .expect("Dixon is decided order-ascending at max_degree = 4");
        println!(
            "dixon (minimal): order {} in {:?} after {} probes",
            sharp.value.result.order,
            start.elapsed(),
            sharp.value.probes
        );
        assert_eq!(sharp.value.result.order, 2);
        assert!(
            sharp.value.order_is_minimal,
            "every order-1 degree in bounds was refused, so order 2 is minimal"
        );
        // All five order-1 degrees, then (2,0) and (2,1).
        assert!(
            sharp.value.probes > opts.max_degree,
            "minimality is bought with the full low-order sweep, got {} probes",
            sharp.value.probes
        );
        // Same relation, sharper claim: the certificate is still verified
        // exactly, so this is not a second, weaker code path.
        assert_eq!(sharp.value.result.coeffs.len(), 3);
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
