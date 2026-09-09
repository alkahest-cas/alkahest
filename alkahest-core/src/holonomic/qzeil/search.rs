//! `q`-Zeilberger: creative telescoping for `q`-hypergeometric terms.
//!
//! Given a `q`-proper hypergeometric `F(n, k)` (see [`super::term`]), this
//! searches for
//!
//! ```text
//! Σ_{i=0}^{J} a_i(qⁿ)·F(n+i, k) = G(n, k+1) − G(n, k),   G(n,k) = R(qⁿ, q^k)·F(n,k)
//! ```
//!
//! with `a_i ∈ Q(q)[x]`, `x = qⁿ`, and an exact rational certificate
//! `R ∈ Q(q)(x)(y)`, `y = q^k`.
//!
//! # Method — the classical one, read in the `q`-shift
//!
//! Everything in [`mod@super::super::zeilberger`]'s method section applies verbatim
//! once `k ↦ k+1` is read as `y ↦ q·y`, because that is an automorphism of
//! `Q(q)(x)(y)` in exactly the way `k ↦ k+1` is one of `Q(n)(k)`:
//!
//! 1. `p(y) = F(n,k+1)/F(n,k)` and `c_i(y) = F(n+i,k)/F(n,k)`, exactly, from
//!    [`super::term::QProperTerm`].
//! 2. `D(y)`, a common denominator of the `c_i` over `Q(q)(x)[y]`; work with
//!    `W = F/D`, whose shift quotient is `ρ(y) = p(y)·D(y)/D(q·y)`.
//! 3. `q`-Gosper normal form `ρ = A(y)·C(q·y) / (B(y)·C(y))` with
//!    `gcd(A(y), B(q^h·y))` a unit for every `h ≥ 0` — the shifted-gcd loop of
//!    the classical construction with the multiplicative shift substituted for
//!    the additive one.
//! 4. Key equation `A(y)·X(q·y) − B(y/q)·X(y) = C(y)·N(y)`, linear over `Q(q)(x)`
//!    in the unknowns `{a_i}` and the coefficients of `X`, with
//!    `R = B(y/q)·X(y) / (C(y)·D(y))`.
//! 5. Solve over the field `Q(q)(x)`; clear denominators into `Q(q)[x]`,
//!    rescaling `R` by the same `y`-free factor.
//! 6. **Re-verify the candidate exactly** as an identity in `Q(q)(x)(y)` —
//!    `Σ_i a_i·c_i = R(q·y)·p(y) − R(y)` — and return it only then. A candidate
//!    that fails is discarded and the search continues; an unverified
//!    certificate is never returned. This is the same non-negotiable discipline
//!    as the classical module's, and it is the only thing that makes a returned
//!    result a proof.

use super::field::{
    clear_denominators_x, clear_field_refusal, qq_pow, ratx_terms, take_field_refusal,
    FieldRefusal, PolyX, PolyY, RatX, RatY, MAX_FIELD_ELEMENT_TERMS,
};
use super::term::QProperTerm;
use super::QHolonomicError;
use crate::holonomic::hyperterm::rn_to_expr;
use crate::holonomic::qfield::{
    clear_denominators, clear_gcd_stop, enter_gcd_work_scope, rn_div, rn_is_zero, rn_poly,
    take_gcd_stop, GcdStop, Rn,
};
use crate::holonomic::zeilberger::OrderSearch;
use crate::kernel::{ExprId, ExprPool};

// ---------------------------------------------------------------------------
// Resource ceilings
// ---------------------------------------------------------------------------
//
// `telescope_md` has had ceilings since it was written; this module was
// written *after* the 2026-08-13 report on `zeilberger`'s unbounded search and
// still had none, so at its own documented defaults a class-legal summand
// (`Σ_k [2n;k]_q`) ran for eight minutes with no output and had to be killed.
// The two below bound the two things that can grow, and they are deliberately
// different in kind, because the cost of this search is fragile in the *input*
// and only partly in the knobs:
//
//   * the **shape** of the linear system — equations × unknowns — which
//     `max_order`/`max_degree` already influence but do not bound, because the
//     equation count comes from the degree of the key equation, not from the
//     caller's `max_degree`; and
//   * the **size of the numbers** in it, which nothing about the shape
//     predicts. `Σ_k [n;k]_q` and `Σ_k [2n;k]_q` present systems of similar
//     shape; only the second one's coefficients explode.
//
// Both refuse a probe *before* it is attempted (or, for the size ceiling, the
// moment an entry crosses the line) and are reported in the exhaustion message
// so that a refusal is never mistaken for "the grid was covered and nothing
// was found" — see `SearchExhausted`'s text at the end of
// `q_zeilberger_on_term`.

/// Largest linear system, in cells (equations × unknowns), that one
/// `(order, degree)` probe may assemble.
///
/// The system is `n_eq × n_var` over `Q(q)(x)`, and elimination is
/// `O(n_eq · n_var²)` *field* operations, each of which is a rational-function
/// arithmetic operation, not a machine one. 4 000 cells covers every probe the
/// module's own test suite makes by a wide margin (the largest is under 400).
const MAX_SYSTEM_CELLS: usize = 4_000;

/// Largest total across every probe of one search call, so that a caller
/// cannot pay the per-probe ceiling once for each of `max_order × max_degree`
/// combinations.
const MAX_CUMULATIVE_SYSTEM_CELLS: usize = 20_000;

/// The outcome of one `(order, degree)` probe.
enum Probe {
    /// `(X coefficients, a_0..a_{order−1})`.
    Solved(Vec<RatX>, Vec<RatX>),
    /// The system has no solution of this shape — an ordinary miss.
    NoSolution,
    /// Refused by a resource ceiling before (or during) the solve. Distinct
    /// from [`Probe::NoSolution`] on purpose: it must not be reported as
    /// evidence that no certificate of this shape exists.
    Refused,
}

/// Everything [`super::q_zeilberger`] takes beyond the term itself.
///
/// `max_order` and `max_degree` are upper **bounds**: the `(order, degree)`
/// grid is walked by iterative deepening, cheapest probe first, so raising
/// either only widens what can be found. The defaults are lower than the
/// classical module's because the coefficient field is one level taller —
/// every `Q(q)(x)` pivot is itself a quotient of polynomials in `q` — and a
/// degree-16 sweep in `y` is not a computation anyone is waiting for.
#[derive(Debug, Clone, Copy)]
pub struct QZeilbergerOpts {
    /// Largest recurrence order `J` to try.
    pub max_order: usize,
    /// Largest certificate-polynomial degree (in `y`) to try, per order.
    pub max_degree: usize,
    /// How the `(order, degree)` grid is traversed; only
    /// [`OrderSearch::MinimalOrder`] can establish minimality, and it pays the
    /// whole low-order sweep for it.
    pub search: OrderSearch,
    /// The smallest `n` the boundary verdict is asserted for. It is part of the
    /// verdict, not of the search, and it is reported back in
    /// [`super::QBoundaryStatus::side_conditions`] rather than assumed.
    pub n_min: i64,
}

impl Default for QZeilbergerOpts {
    fn default() -> Self {
        QZeilbergerOpts {
            max_order: 3,
            max_degree: 6,
            search: OrderSearch::CostOrdered,
            n_min: 0,
        }
    }
}

/// A **verified** `q`-Zeilberger certificate.
#[derive(Debug, Clone)]
pub struct QZeilbergerResult {
    /// Recurrence order `J`; `coeffs.len() == order + 1`.
    pub order: usize,
    /// `a_0, …, a_J` as expressions in `q` and `n` (through `q^n`).
    pub coeffs: Vec<ExprId>,
    /// `R` as an expression in `q`, `n`, `k` (through `q^n`, `q^k`).
    pub certificate: ExprId,
    /// The same coefficients as elements of `Q(q)[x]`, for the boundary
    /// analysis and for re-checking against exact `q`-series terms.
    pub coeffs_x: Vec<PolyX>,
    /// The certificate as an element of `Q(q)(x)(y)`.
    pub certificate_xy: RatY,
}

/// [`QZeilbergerResult`] plus what the search that produced it established.
#[derive(Debug, Clone)]
pub struct QZeilbergerReport {
    pub result: QZeilbergerResult,
    /// `true` only when every lower order was refused at every degree in
    /// bounds — never inferred from the traversal mode.
    pub order_is_minimal: bool,
    /// How many `(order, degree)` probes were made, the successful one included.
    pub probes: usize,
}

/// `y^j` as an element of `Q(q)(x)[y]`.
fn y_mono(j: usize) -> PolyY {
    let mut coeffs = vec![RatX::zero(); j + 1];
    coeffs[j] = RatX::one();
    PolyY::from_coeffs(coeffs)
}

/// The `q`-analogue of Gosper's normal form: `p/r = A(y)·C(q·y) / (B(y)·C(y))`
/// with `gcd(A(y), B(q^h·y))` a unit for every `h ≥ 0`.
fn q_gosper_normal_form(mut p: PolyY, mut r: PolyY) -> Option<(PolyY, PolyY, PolyY)> {
    if p.is_zero() {
        return Some((PolyY::zero(), PolyY::one(), PolyY::one()));
    }
    if r.is_zero() {
        return None;
    }
    let lc_p = p.leading_coeff();
    let lc_r = r.leading_coeff();
    let z_scale = lc_p.div(&lc_r)?;
    p = p.scale(&lc_p.inv()?);
    r = r.scale(&lc_r.inv()?);

    let mut a = p;
    let mut b = r;
    let mut c = PolyY::one();
    let bound = (a.degree().max(0) + b.degree().max(0)).max(1) as usize + 16;

    loop {
        let mut found = false;
        // `h = 0` included, for the same reason as in the classical module: a
        // plain common factor left in both `A` and `B` breaks the ansatz.
        for h in 0..=bound {
            let bshift = b.qshift_y(h as i64);
            let d = PolyY::gcd(&a, &bshift);
            if d.is_zero() || d.degree() == 0 {
                continue;
            }
            let Some(an) = PolyY::exact_div(&a, &d) else {
                continue;
            };
            let Some(bn) = PolyY::exact_div(&b, &d.qshift_y(-(h as i64))) else {
                continue;
            };
            a = an;
            b = bn;
            let mut prod = PolyY::one();
            for j in 1..=h {
                prod = prod.mul(&d.qshift_y(-(j as i64)));
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

/// Gaussian elimination over the field `Q(q)(x)`.
///
/// `Ok(None)` is an ordinary "no solution"; `Err` is a refusal — either an
/// active [`crate::budget`] (wall clock, steps, memory) or
/// [`MAX_FIELD_ELEMENT_TERMS`], the ceiling on how big one field element may
/// grow. Both must stay distinguishable from "no solution" all the way out.
fn field_solve(
    mut mat: Vec<Vec<RatX>>,
    mut rhs: Vec<RatX>,
) -> Result<Option<Vec<RatX>>, QHolonomicError> {
    let nrows = mat.len();
    if nrows == 0 {
        return Ok(Some(vec![]));
    }
    let ncols = mat[0].len();
    let mut row = 0;
    for col in 0..ncols {
        if row >= nrows {
            break;
        }
        checkpoint()?;
        let Some(pr) = (row..nrows).find(|&r| !mat[r][col].is_zero()) else {
            continue;
        };
        mat.swap(row, pr);
        rhs.swap(row, pr);
        let Some(inv) = mat[row][col].inv() else {
            return Ok(None);
        };
        for entry in mat[row].iter_mut().skip(col) {
            *entry = entry.mul(&inv);
        }
        rhs[row] = rhs[row].mul(&inv);
        let pivot_row = mat[row].clone();
        let pivot_rhs = rhs[row].clone();
        for r in 0..nrows {
            if r == row {
                continue;
            }
            let v = mat[r][col].clone();
            if v.is_zero() {
                continue;
            }
            // Per row, not per pivot: it is the entries, not the shape, that
            // grow here.
            checkpoint()?;
            for (entry, pivot) in mat[r].iter_mut().zip(pivot_row.iter()).skip(col) {
                *entry = entry.sub(&pivot.mul(&v));
                if ratx_terms(entry) > MAX_FIELD_ELEMENT_TERMS {
                    return Err(size_ceiling_error(ratx_terms(entry)));
                }
            }
            rhs[r] = rhs[r].sub(&pivot_rhs.mul(&v));
        }
        row += 1;
    }
    for (r, mrow) in mat.iter().enumerate() {
        if mrow.iter().all(RatX::is_zero) && !rhs[r].is_zero() {
            return Ok(None);
        }
    }
    let mut sol = vec![RatX::zero(); ncols];
    for r in (0..nrows).rev() {
        if let Some(j) = mat[r].iter().position(|e| !e.is_zero()) {
            checkpoint()?;
            let mut sum = rhs[r].clone();
            for cidx in (j + 1)..ncols {
                sum = sum.sub(&mat[r][cidx].mul(&sol[cidx]));
            }
            let Some(q) = sum.div(&mat[r][j]) else {
                return Ok(None);
            };
            sol[j] = q;
        }
    }
    Ok(Some(sol))
}

/// Turn a [`crate::budget`] trip into this module's exhaustive error type,
/// recording the real cause out of band for the bindings to raise as
/// `BudgetExceededError` — see [`crate::budget::record_trip`].
fn trip_to_error(trip: crate::budget::BudgetTrip) -> QHolonomicError {
    use crate::errors::AlkahestError;
    crate::budget::record_trip(trip);
    QHolonomicError::SearchExhausted(format!(
        "the q-Zeilberger search was stopped before it finished, and no certificate was found \
         up to that point — this is NOT a statement that none exists: {} [{}]",
        trip,
        trip.code()
    ))
}

/// Cooperative checkpoint: wall clock, steps, cancellation, and the memory
/// ceilings of [`crate::budget::memory`]. Before this existed, `q_zeilberger`
/// honoured no budget at all — `Budget(wall_ms=...)` did not stop it.
fn checkpoint() -> Result<(), QHolonomicError> {
    crate::budget::check_all().map_err(trip_to_error)
}

/// Marker error for a [`MAX_FIELD_ELEMENT_TERMS`] trip.
///
/// Carried as `SearchExhausted` because `QHolonomicError` is public and
/// exhaustive; the caller recognises it by [`is_size_ceiling`] and reports the
/// probe as [`Probe::Refused`], never as "no solution".
fn size_ceiling_error(terms: usize) -> QHolonomicError {
    QHolonomicError::SearchExhausted(format!(
        "{SIZE_CEILING_TAG}: a coefficient of the linear system reached {terms} rational \
         numbers, past this module's MAX_FIELD_ELEMENT_TERMS = {MAX_FIELD_ELEMENT_TERMS}"
    ))
}

const SIZE_CEILING_TAG: &str = "q-Zeilberger field-element ceiling";

fn is_size_ceiling(e: &QHolonomicError) -> bool {
    matches!(e, QHolonomicError::SearchExhausted(m) if m.starts_with(SIZE_CEILING_TAG))
}

/// Solve `A(y)·X(q·y) − B(y/q)·X(y) = C(y)·N(y)` for a degree-`d` `X` and the
/// coefficients `a_0..a_{order−1}` (with `a_order` normalised to `1`).
fn try_solve(
    aa: &PolyY,
    b_eq: &PolyY,
    c_ci: &[PolyY],
    order: usize,
    d: usize,
    cumulative_cells: &mut usize,
) -> Result<Probe, QHolonomicError> {
    let mut bx: Vec<PolyY> = Vec::with_capacity(d + 1);
    for j in 0..=d {
        let yj = y_mono(j);
        // `X(q·y)`'s `j`-th basis element is `q^j·y^j`.
        let shifted = yj.scale(&RatX::from_rn(qq_pow(j as i64)));
        bx.push(aa.mul(&shifted).sub(&b_eq.mul(&yj)));
    }

    let mut max_deg = 0i32;
    for p in bx.iter().chain(c_ci.iter()) {
        max_deg = max_deg.max(p.degree());
    }
    let n_eq = (max_deg.max(0) as usize) + 1;
    let n_var = (d + 1) + order;

    // Shape ceilings, both purely arithmetic and both *before* the system is
    // assembled: `n_eq` comes from the degree of the key equation, so it is
    // not bounded by the caller's `max_degree`.
    let cells = n_eq.saturating_mul(n_var);
    if cells > MAX_SYSTEM_CELLS {
        return Ok(Probe::Refused);
    }
    if cumulative_cells.saturating_add(cells) > MAX_CUMULATIVE_SYSTEM_CELLS {
        return Ok(Probe::Refused);
    }
    *cumulative_cells += cells;

    let mut mat = vec![vec![RatX::zero(); n_var]; n_eq];
    let mut rhs = vec![RatX::zero(); n_eq];
    for (m, row) in mat.iter_mut().enumerate() {
        for (j, bxj) in bx.iter().enumerate() {
            row[j] = bxj.coeff(m);
        }
        for i in 0..order {
            row[(d + 1) + i] = c_ci[i].coeff(m).neg();
        }
        rhs[m] = c_ci[order].coeff(m);
    }

    let sol = match field_solve(mat, rhs) {
        Ok(Some(sol)) => sol,
        Ok(None) => return Ok(Probe::NoSolution),
        Err(e) if is_size_ceiling(&e) => return Ok(Probe::Refused),
        Err(e) => return Err(e),
    };
    Ok(Probe::Solved(sol[..=d].to_vec(), sol[(d + 1)..].to_vec()))
}

/// Rescale a `Q(q)[x]` family by one common element of `Q(q)` so that every
/// coefficient is an integer polynomial in `q` with overall content 1.
///
/// Returns the rescaled family and the scale, or `None` when the family is
/// identically zero (nothing to normalise against).
fn primitive_family(family: &[PolyX]) -> Option<(Vec<PolyX>, Rn)> {
    let flat: Vec<Rn> = family
        .iter()
        .flat_map(|p| p.coeffs.iter().cloned())
        .collect();
    let cleared = clear_denominators(&flat);
    // The scale is common to the whole family, so any non-zero entry recovers it.
    let idx = flat.iter().position(|c| !rn_is_zero(c))?;
    let scale = rn_div(&rn_poly(cleared[idx].clone()), &flat[idx])?;
    let mut out = Vec::with_capacity(family.len());
    let mut at = 0usize;
    for p in family {
        let len = p.coeffs.len();
        out.push(PolyX::from_coeffs(
            cleared[at..at + len].iter().cloned().map(rn_poly).collect(),
        ));
        at += len;
    }
    Some((out, scale))
}

/// Degree-independent setup for one recurrence order.
struct OrderState {
    c: Vec<RatY>,
    dden: PolyY,
    aa: PolyY,
    b_eq: PolyY,
    cc: PolyY,
    c_ci: Vec<PolyY>,
}

fn order_state(
    f: &QProperTerm,
    p: &RatY,
    order: usize,
) -> Result<Option<OrderState>, QHolonomicError> {
    let c: Vec<RatY> = (0..=order as i64)
        .map(|i| f.ratio_n(i))
        .collect::<Result<_, _>>()?;

    let mut dden = PolyY::one();
    for ci in &c {
        dden = PolyY::lcm(&dden, &ci.den);
    }
    if dden.is_zero() {
        return Ok(None);
    }
    let ci_polys: Option<Vec<PolyY>> = c
        .iter()
        .map(|ci| PolyY::exact_div(&dden.mul(&ci.num), &ci.den))
        .collect();
    let Some(ci_polys) = ci_polys else {
        return Ok(None);
    };

    // ρ(y) = p(y)·D(y)/D(q·y).
    let rho_num = p.num.mul(&dden);
    let rho_den = p.den.mul(&dden.qshift_y(1));
    let Some((aa, bb, cc)) = q_gosper_normal_form(rho_num, rho_den) else {
        return Ok(None);
    };
    let b_eq = bb.qshift_y(-1);
    let c_ci: Vec<PolyY> = ci_polys.iter().map(|q| cc.mul(q)).collect();

    Ok(Some(OrderState {
        c,
        dden,
        aa,
        b_eq,
        cc,
        c_ci,
    }))
}

/// See the classical module: one extra order costs about what three extra
/// certificate degrees cost, so the cost-ordered plan sweeps
/// `3·(order−1) + d = t`.
const ORDER_COST_IN_DEGREE_STEPS: usize = 3;

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

/// `q`-Zeilberger's algorithm on an already-parsed term.
pub fn q_zeilberger_on_term(
    f: &QProperTerm,
    q: ExprId,
    n: ExprId,
    k: ExprId,
    pool: &ExprPool,
    opts: &QZeilbergerOpts,
) -> Result<QZeilbergerReport, QHolonomicError> {
    if opts.max_order == 0 || opts.max_degree == 0 {
        return Err(QHolonomicError::InvalidInput(
            "max_order and max_degree must both be at least 1".into(),
        ));
    }
    // Only this call's trip may be attributed to this call.
    crate::budget::clear_trip();
    // The `Z[q][x]` gcd under every `Q(q)(x)` operation is bounded for the
    // whole search, not per probe: once the ceiling is reached, the remaining
    // probes refuse immediately instead of each paying it again.
    let _gcd_scope = enter_gcd_work_scope();
    let p = f.ratio_k()?;

    let mut states: Vec<Option<OrderState>> = Vec::with_capacity(opts.max_order);
    let mut degrees_failed = vec![0usize; opts.max_order];
    let mut cumulative_cells: usize = 0;
    let mut refused_by_ceiling = false;
    // `PolyY::gcd` (via `RatY::normalize`) is infallible by signature, so it
    // reports a ceiling or budget stop out of band; this expands to the `?`,
    // `continue` and flag update that every heavy step below needs, without
    // threading a `Result` through the whole coefficient tower.
    macro_rules! bail_on_field_refusal {
        () => {
            match take_field_refusal() {
                None => {}
                Some(FieldRefusal::Budget(t)) => return Err(trip_to_error(t)),
                Some(FieldRefusal::SizeCeiling(_)) => {
                    refused_by_ceiling = true;
                    continue;
                }
            }
            match take_gcd_stop() {
                None => {}
                Some(GcdStop::Budget(t)) => return Err(trip_to_error(t)),
                Some(GcdStop::Size(_)) | Some(GcdStop::Work(_)) => {
                    refused_by_ceiling = true;
                    continue;
                }
            }
        };
    }
    for (order, d) in search_plan(opts.max_order, opts.max_degree, opts.search) {
        degrees_failed[order - 1] += 1;
        checkpoint()?;
        clear_field_refusal();
        clear_gcd_stop();
        while states.len() < order {
            states.push(order_state(f, &p, states.len() + 1)?);
        }
        let Some(state) = &states[order - 1] else {
            continue;
        };
        let probe = try_solve(
            &state.aa,
            &state.b_eq,
            &state.c_ci,
            order,
            d,
            &mut cumulative_cells,
        )?;
        // A ceiling that fired inside the solve makes this probe a refusal,
        // not a miss — checked before the `NoSolution` arm can swallow it.
        bail_on_field_refusal!();
        let (x_coeffs, lam_below) = match probe {
            Probe::Solved(x, lam) => (x, lam),
            Probe::NoSolution => continue,
            Probe::Refused => {
                refused_by_ceiling = true;
                continue;
            }
        };

        let mut lam_full = lam_below;
        lam_full.push(RatX::one()); // a_order = 1

        let x_poly = PolyY::from_coeffs(x_coeffs);
        let r_pre = RatY {
            num: state.b_eq.mul(&x_poly),
            den: state.cc.mul(&state.dden),
        }
        .normalize();

        bail_on_field_refusal!();
        let (a_int, scale) = clear_denominators_x(&lam_full);
        if a_int.iter().all(PolyX::is_zero) || a_int[order].is_zero() {
            continue;
        }
        // Multiplying the whole identity by the `y`-free `scale` keeps it exact.
        let mut r_final = RatY {
            num: r_pre.num.scale(&scale),
            den: r_pre.den.clone(),
        }
        .normalize();

        // Second normalisation, cosmetic but worth it: pull the `Q(q)`
        // denominators and the integer content out of the *whole* family at
        // once, so the coefficients come back as polynomials in `q` and `qⁿ`
        // (`q^{n+1} − 1`) rather than as quotients (`qⁿ − 1/q`). The scale is
        // one common element of `Q(q)`, and it multiplies the certificate too,
        // so the identity is untouched — and it is re-verified below either way.
        bail_on_field_refusal!();
        let a_int = primitive_family(&a_int)
            .map(|(family, s)| {
                r_final = r_final.mul(&RatY::from_ratx(RatX::from_rn(s)));
                family
            })
            .unwrap_or(a_int);

        // § non-negotiable discipline: an exact identity in Q(q)(x)(y), or no
        // result at all.
        let mut lhs = RatY::zero();
        for (i, ci) in state.c.iter().enumerate() {
            lhs = lhs.add(&RatY::from_ratx(RatX::from_poly(a_int[i].clone())).mul(ci));
        }
        bail_on_field_refusal!();
        let rhs_check = r_final.qshift_y(1).mul(&p).sub(&r_final);
        if !lhs.sub(&rhs_check).is_zero() {
            continue;
        }

        // Rendered forms are simplified once, here: the builders emit
        // `1*q^n + -1` shapes that are correct but unreadable, and a caller
        // reading `a_1` off a returned certificate should not have to.
        bail_on_field_refusal!();
        let simp = |e: ExprId| crate::simplify::simplify(e, pool).value;
        let coeffs: Vec<ExprId> = a_int
            .iter()
            .map(|c| simp(polyx_to_expr(pool, q, n, c)))
            .collect();
        let certificate = simp(raty_to_expr(pool, q, n, k, &r_final));
        let order_is_minimal = (1..order).all(|j| degrees_failed[j - 1] == opts.max_degree + 1);

        return Ok(QZeilbergerReport {
            result: QZeilbergerResult {
                order,
                coeffs,
                certificate,
                coeffs_x: a_int,
                certificate_xy: r_final,
            },
            order_is_minimal,
            probes: degrees_failed.iter().sum(),
        });
    }

    let ceiling_note = if refused_by_ceiling {
        format!(
            " (at least one (order, degree) combination within these bounds was refused by this \
             module's resource ceilings rather than attempted — MAX_SYSTEM_CELLS = \
             {MAX_SYSTEM_CELLS} equations x unknowns for any single probe, \
             MAX_CUMULATIVE_SYSTEM_CELLS = {MAX_CUMULATIVE_SYSTEM_CELLS} across the whole \
             search, MAX_FIELD_ELEMENT_TERMS = {MAX_FIELD_ELEMENT_TERMS} rational numbers in \
             any one coefficient of the linear system; so this is NOT a statement that no \
             q-recurrence exists within these bounds)"
        )
    } else {
        String::new()
    };
    Err(QHolonomicError::SearchExhausted(format!(
        "no verified q-recurrence of order <= {} with certificate degree <= {} in q^k was \
         found{ceiling_note}",
        opts.max_order, opts.max_degree
    )))
}

// ---------------------------------------------------------------------------
// Rendering back into expressions
// ---------------------------------------------------------------------------

/// `q^{e·v}` as an expression, for the substitutions `x = q^n`, `y = q^k`.
fn q_pow_var(pool: &ExprPool, q: ExprId, v: ExprId, e: usize) -> Option<ExprId> {
    match e {
        0 => None,
        1 => Some(pool.pow(q, v)),
        _ => Some(pool.pow(q, pool.mul(vec![pool.integer(e as i64), v]))),
    }
}

/// An element of `Q(q)` as an expression in `q`.
///
/// `hyperterm::rn_to_expr` renders a one-variable rational function against
/// whichever symbol it is handed; here that symbol is `q` rather than `n`.
fn qq_to_expr(pool: &ExprPool, q: ExprId, c: &Rn) -> ExprId {
    rn_to_expr(pool, q, c)
}

/// An element of `Q(q)[x]` as an expression in `q` and `n`.
pub fn polyx_to_expr(pool: &ExprPool, q: ExprId, n: ExprId, p: &PolyX) -> ExprId {
    let mut terms = Vec::new();
    for (deg, c) in p.coeffs.iter().enumerate() {
        if rn_is_zero(c) {
            continue;
        }
        let ce = qq_to_expr(pool, q, c);
        terms.push(match q_pow_var(pool, q, n, deg) {
            None => ce,
            Some(xd) => pool.mul(vec![ce, xd]),
        });
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

fn polyy_to_expr(pool: &ExprPool, q: ExprId, n: ExprId, k: ExprId, p: &PolyY) -> ExprId {
    let mut terms = Vec::new();
    for (deg, c) in p.coeffs.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let ce = ratx_to_expr(pool, q, n, c);
        terms.push(match q_pow_var(pool, q, k, deg) {
            None => ce,
            Some(yd) => pool.mul(vec![ce, yd]),
        });
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

/// An element of `Q(q)(x)` as an expression in `q` and `n`.
pub fn ratx_to_expr(pool: &ExprPool, q: ExprId, n: ExprId, r: &RatX) -> ExprId {
    let num = polyx_to_expr(pool, q, n, &r.num);
    if r.den.eq_poly(&PolyX::one()) {
        return num;
    }
    let den = polyx_to_expr(pool, q, n, &r.den);
    pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))])
}

/// An element of `Q(q)(x)(y)` as an expression in `q`, `n` and `k`.
pub fn raty_to_expr(pool: &ExprPool, q: ExprId, n: ExprId, k: ExprId, r: &RatY) -> ExprId {
    let num = polyy_to_expr(pool, q, n, k, &r.num);
    if r.den.eq_poly(&PolyY::one()) {
        return num;
    }
    let den = polyy_to_expr(pool, q, n, k, &r.den);
    pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))])
}
