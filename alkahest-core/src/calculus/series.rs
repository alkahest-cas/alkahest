//! Truncated Taylor / Laurent series with symbolic [`crate::kernel::ExprData::BigO`] remainder (V2-15).

use crate::budget::BudgetError;
use crate::diff::{diff, DiffError};
use crate::flint::FlintPoly;
use crate::kernel::{subs, Domain, ExprData, ExprId, ExprPool};
use crate::poly::{RationalFunction, UniPoly};
use crate::simplify::simplify;
use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of [`series`] — truncated expansion plus big-O bound as one [`ExprId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Series(pub ExprId);

impl Series {
    pub fn expr(self) -> ExprId {
        self.0
    }
}

#[derive(Debug)]
pub enum SeriesError {
    /// Differentiation failed while forming Taylor coefficients.
    Diff(DiffError),
    /// The requested `order` is not one this call can expand to: it was `0`,
    /// or the expansion ran past the work ceiling / an active
    /// [`crate::budget`] before reaching it.
    ///
    /// The second reading is the carrier for a *refusal* — see
    /// [`take_series_refusal`] for which of the two happened, and
    /// [`SeriesRefusal`] for why the refusal cannot be its own variant.
    InvalidOrder,
}

impl fmt::Display for SeriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeriesError::Diff(e) => write!(f, "{e}"),
            SeriesError::InvalidOrder => write!(
                f,
                "series order must be >= 1 and reachable: the expansion is not \
                 available at the order requested"
            ),
        }
    }
}

impl std::error::Error for SeriesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SeriesError::Diff(e) => Some(e),
            SeriesError::InvalidOrder => None,
        }
    }
}

impl crate::errors::AlkahestError for SeriesError {
    fn code(&self) -> &'static str {
        match self {
            SeriesError::Diff(_) => "E-SERIES-001",
            SeriesError::InvalidOrder => "E-SERIES-002",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            SeriesError::Diff(_) => {
                Some("ensure all functions are registered primitives with differentiation rules")
            }
            SeriesError::InvalidOrder => Some(
                "pass order >= 1 (exclusive truncation degree in x); if the order was \
                 already positive the expansion exceeded the work ceiling — ask for a \
                 lower order, or simplify the expression so its derivatives close",
            ),
        }
    }
}

impl From<DiffError> for SeriesError {
    fn from(e: DiffError) -> Self {
        SeriesError::Diff(e)
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Truncated Taylor or Laurent expansion of `expr` in `var` about `point`.
///
/// Let `h = var - point`. The returned expression has the shape
/// `⋯ + O(h^k)` where `k = order` for analytic series (`valuation ≥ 0`), and
/// `k = 1` when a polar term (`valuation < 0`) is present — matching the
/// Laurent examples in the roadmap (`1/x` about `0` gives `x⁻¹ + O(x)`).
///
/// The `order` parameter matches the Taylor convention used in the roadmap:
/// include powers `h^e` with `valuation ≤ e < order` when `valuation ≥ 0`, and
/// when `valuation < 0` include the polar tail using `order` Taylor coefficients
/// of the analytic factor `h^{-valuation} · f`.
///
/// # Termination
///
/// The coefficient loop is bounded: it honours [`crate::budget`] (wall clock,
/// steps, [`crate::budget::request_cancel`]) and, with no budget active, an
/// internal work ceiling ([`MAX_SERIES_POOL_GROWTH`]). Coefficients are formed
/// by repeated differentiation *without* re-simplifying, so an expression whose
/// derivatives do not close — `√(t⁻² + t⁻¹)` is the standard example — grows by
/// a constant factor per coefficient and order 32 is not slow but unreachable.
///
/// Running out of room is reported as **`Err(SeriesError::InvalidOrder)` with a
/// [`take_series_refusal`] pending**, never as a shorter series: a truncated
/// expansion still labelled `O(hᵒʳᵈᵉʳ)` would be a false statement about the
/// remainder, and that is a lie where a refusal is merely a limitation.
pub fn series(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<Series, SeriesError> {
    let frame = enter_series_frame();
    // The ceiling is what makes the loop stoppable at all: `local_expansion` is
    // one uninterruptible call from here, so there is nowhere else to put a
    // checkpoint. Unlike `limit`'s, this one refuses instead of settling for
    // the prefix it managed to compute.
    let _ceiling = enter_coeff_ceiling(pool.len().saturating_add(MAX_SERIES_POOL_GROWTH));

    let LocalExpansion {
        valuation,
        coeffs,
        h_expr,
    } = local_expansion(expr, var, point, order, pool)?;

    if frame.refusal_pending() {
        return Err(SeriesError::InvalidOrder);
    }

    Ok(assemble_series(&coeffs, valuation, h_expr, order, pool))
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Local Laurent / Taylor data about `point`: `expr = ∑ᵢ coeffᵢ · h^{valuation+i}` up to truncation.
///
/// `h` is `var - point`, or bare `var` when `point` is the integer zero (matching [`series`]).
#[derive(Clone, Debug)]
pub(crate) struct LocalExpansion {
    pub valuation: i32,
    pub coeffs: Vec<ExprId>,
    pub h_expr: ExprId,
}

pub(crate) fn local_expansion(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<LocalExpansion, SeriesError> {
    if order == 0 {
        return Err(SeriesError::InvalidOrder);
    }

    let xi = pool.symbol("__sxp", Domain::Real);
    let mut map = HashMap::new();
    map.insert(var, pool.add(vec![point, xi]));
    let shifted = subs(expr, &map, pool);

    let h_expr = expansion_increment(pool, var, point);

    expansion_matched_laurent(shifted, xi, h_expr, order, pool)
}

fn factorial_u32(n: u32) -> rug::Integer {
    let mut r = rug::Integer::from(1);
    for i in 2..=n {
        r *= i;
    }
    r
}

fn expansion_increment(pool: &ExprPool, var: ExprId, point: ExprId) -> ExprId {
    match pool.get(point) {
        ExprData::Integer(n) if n.0 == 0 => var,
        _ => pool.add(vec![var, pool.mul(vec![pool.integer(-1_i32), point])]),
    }
}

fn laurent_big_o_pow(valuation: i32, order: u32) -> i64 {
    if valuation < 0 {
        1
    } else {
        order as i64
    }
}

fn is_structural_zero(id: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(id), ExprData::Integer(n) if n.0 == 0)
}

fn collect_atom_factors(expr: ExprId, pool: &ExprPool) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let n = pool.with(exp, |d| match d {
                ExprData::Integer(i) => Some(i.0.clone()),
                _ => None,
            })?;
            if n > 0 {
                Some((vec![expr], vec![]))
            } else if n < 0 {
                let mag = (-n).to_u32()?;
                let pos_exp = pool.integer(mag as i64);
                Some((vec![], vec![pool.pow(base, pos_exp)]))
            } else {
                Some((vec![pool.integer(1_i32)], vec![]))
            }
        }
        ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_)
        | ExprData::Symbol { .. }
        | ExprData::Func { .. } => Some((vec![expr], vec![])),
        ExprData::Add(_)
        | ExprData::Mul(_)
        | ExprData::Piecewise { .. }
        | ExprData::Predicate { .. }
        | ExprData::Forall { .. }
        | ExprData::Exists { .. }
        | ExprData::RootSum { .. }
        | ExprData::BigO(_) => None,
    }
}

fn collect_term_factors(expr: ExprId, pool: &ExprPool) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    match pool.get(expr) {
        ExprData::Mul(args) => {
            let mut nums = Vec::new();
            let mut dens = Vec::new();
            for &a in &args {
                let (n, d) = collect_atom_factors(a, pool)?;
                nums.extend(n);
                dens.extend(d);
            }
            Some((nums, dens))
        }
        _ => collect_atom_factors(expr, pool),
    }
}

fn product_sorted(pool: &ExprPool, factors: Vec<ExprId>) -> ExprId {
    match factors.len() {
        0 => pool.integer(1_i32),
        1 => factors[0],
        _ => pool.mul(factors),
    }
}

fn unipoly_valuation(p: &UniPoly) -> Option<u32> {
    for (i, c) in p.coefficients().into_iter().enumerate() {
        if c != 0 {
            return Some(i as u32);
        }
    }
    None
}

fn unipoly_strip_low(p: &UniPoly, k: u32) -> UniPoly {
    let coeffs: Vec<rug::Integer> = p.coefficients().into_iter().skip(k as usize).collect();
    UniPoly {
        var: p.var,
        coeffs: FlintPoly::from_rug_coefficients(&coeffs),
    }
}

// ---------------------------------------------------------------------------
// Coefficient-loop ceiling
// ---------------------------------------------------------------------------

/// How many *new* expression nodes one top-level [`series`] call may intern
/// before it refuses.
///
/// Measured rather than guessed, with an order of magnitude of headroom: the
/// heaviest expansions in the Rust and Python suites intern a few thousand nodes
/// (`sin` at order 24: 125; `√(1+x)` at order 24: 677; `tan` at order 16: 1 564;
/// `log(1+x)/(1−x)` at order 20: 4 579), while `√(t⁻² + t⁻¹)` at order 32 doubles
/// per coefficient and reaches this ceiling in a fraction of a second.
///
/// Counting interned nodes rather than iterations catches the pathology directly
/// (it is *size* that explodes, not the iteration count), costs `O(1)` per check
/// — [`ExprPool::len`] is a lock-free counter — and is monotone, so no path can
/// evade it.
pub const MAX_SERIES_POOL_GROWTH: usize = 50_000;

thread_local! {
    /// Absolute `pool.len()` ceiling for [`taylor_coefficients`], or `None` for
    /// "compute every coefficient that was asked for".
    static COEFF_POOL_CEILING: Cell<Option<usize>> = const { Cell::new(None) };
    /// `true` while a [`series`] call is on the stack, which is the only
    /// context in which a truncated coefficient loop is a refusal rather than
    /// the requested behaviour.
    static IN_SERIES: Cell<bool> = const { Cell::new(false) };
    /// The refusal behind the [`SeriesError::InvalidOrder`] the current thread
    /// is about to return, if that error is a work-ceiling trip rather than a
    /// zero `order`.
    static LAST_REFUSAL: Cell<Option<SeriesRefusal>> = const { Cell::new(None) };
}

/// A [`series`] call that could not reach the order it was asked for.
///
/// # Why this is not an error variant
///
/// [`SeriesError`] is a public *exhaustive* enum, so growing it a `Truncated`
/// variant is a major semver break — and so is marking it `#[non_exhaustive]`
/// to allow it later. A correctness fix inside a patch release cannot spend a
/// major version, so the refusal travels out of band: [`series`] returns
/// [`SeriesError::InvalidOrder`], whose reworded text states exactly the
/// disjunction that is known ("the order is not one this call can expand to"),
/// and the real cause is recorded here for [`take_series_refusal`] to hand to
/// the bindings, which raise its own `E-SERIES-003` (or the `E-BUDGET-*` of the
/// budget that tripped).
///
/// This is the pattern [`crate::calculus::limits::last_budget_trip`] uses for
/// budget trips inside `LimitError::DepthExceeded`, and
/// [`crate::matrix::take_zero_test_refusal`] for undecided zero tests inside
/// `MatrixError::SingularMatrix`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeriesRefusal {
    requested: u32,
    computed: u32,
    budget: Option<BudgetError>,
}

impl SeriesRefusal {
    /// Number of Taylor coefficients that were asked for.
    pub fn requested_coefficients(&self) -> u32 {
        self.requested
    }

    /// Number of Taylor coefficients that were formed before the loop stopped.
    ///
    /// Deliberately *not* returned as a series: `assemble_series` would label it
    /// `O(h^requested)`, which is a claim about a remainder nobody bounded.
    pub fn computed_coefficients(&self) -> u32 {
        self.computed
    }

    /// The [`BudgetError`] that stopped this expansion, or `None` when it was
    /// the internal work ceiling.
    pub fn budget(&self) -> Option<BudgetError> {
        self.budget
    }
}

impl fmt::Display for SeriesRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "series expansion stopped after {} of {} Taylor coefficients ({}); \
             refusing to return a shorter series labelled with the requested \
             order, which would understate the O(.) remainder",
            self.computed,
            self.requested,
            match self.budget {
                Some(b) => format!("budget: {b}"),
                None => "internal work ceiling".to_string(),
            }
        )
    }
}

impl std::error::Error for SeriesRefusal {}

impl crate::errors::AlkahestError for SeriesRefusal {
    fn code(&self) -> &'static str {
        "E-SERIES-003"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(
            "ask for a lower order, raise the budget, or rewrite the expression so its \
             repeated derivatives close (nested radicals grow by a constant factor per \
             coefficient)",
        )
    }
}

/// RAII marker for the outermost [`series`] frame on this thread.
pub(crate) struct SeriesFrame {
    outermost: bool,
}

impl SeriesFrame {
    /// Did the coefficient loop stop early during this call?
    fn refusal_pending(&self) -> bool {
        LAST_REFUSAL.with(|c| c.get().is_some())
    }
}

impl Drop for SeriesFrame {
    fn drop(&mut self) {
        if self.outermost {
            IN_SERIES.with(|c| c.set(false));
        }
    }
}

/// Enter a [`series`] frame, clearing any refusal left by an earlier call so a
/// pending one always describes the call that just returned.
fn enter_series_frame() -> SeriesFrame {
    LAST_REFUSAL.with(|c| c.set(None));
    IN_SERIES.with(|c| {
        let already = c.get();
        c.set(true);
        SeriesFrame {
            outermost: !already,
        }
    })
}

/// Take the refusal behind the [`SeriesError::InvalidOrder`] that just came
/// back, if there was one.
///
/// `Some` means the requested order was positive and simply out of reach — the
/// work ceiling or an active [`crate::budget`] stopped the coefficient loop.
/// `None` means the variant means what it has always meant: `order == 0`.
///
/// Consuming, so one refusal is reported once and cannot leak into a later
/// unrelated error. Thread-local, like the ceiling itself.
pub fn take_series_refusal() -> Option<SeriesRefusal> {
    LAST_REFUSAL.with(|c| c.take())
}

/// RAII installer for the [`taylor_coefficients`] ceiling; restores the
/// previous value on drop, including on panic-unwind.
pub(crate) struct CoeffCeiling(Option<usize>);

impl Drop for CoeffCeiling {
    fn drop(&mut self) {
        COEFF_POOL_CEILING.with(|c| c.set(self.0));
    }
}

/// Stop [`taylor_coefficients`] early once the pool has grown past `ceiling`,
/// returning the coefficients computed so far.
///
/// [`crate::calculus::limits`] scans for the first nonzero coefficient, so a
/// short prefix is either enough to answer or an honest "no answer at this
/// order", never a wrong answer, and it simply uses what it got. [`series`]
/// installs a ceiling too — it has to, or the loop is unbounded — but it treats
/// a short prefix as a **refusal** ([`take_series_refusal`]): returning it would
/// understate the `O(·)` term, which would be a lie rather than a limitation.
///
/// Successive Taylor coefficients are formed by differentiating *without*
/// re-simplifying, so for expressions whose derivatives do not close (nested
/// radicals) each one is a constant factor larger than the last. Without this
/// the loop is unbounded in both time and memory, and — being a single call —
/// gives the caller nowhere to place a cancellation checkpoint.
pub(crate) fn enter_coeff_ceiling(ceiling: usize) -> CoeffCeiling {
    COEFF_POOL_CEILING.with(|c| {
        let prev = c.get();
        c.set(Some(ceiling));
        CoeffCeiling(prev)
    })
}

/// `true` when the installed ceiling has been reached, or the ambient
/// [`crate::budget`] has been exhausted / cancelled.
fn coeff_loop_should_stop(pool: &ExprPool) -> bool {
    match COEFF_POOL_CEILING.with(|c| c.get()) {
        Some(ceiling) => pool.len() > ceiling || crate::budget::check().is_err(),
        None => false,
    }
}

fn taylor_coefficients(
    mut cur: ExprId,
    xi: ExprId,
    num: u32,
    pool: &ExprPool,
) -> Result<Vec<ExprId>, SeriesError> {
    let mut mapping = HashMap::new();
    mapping.insert(xi, pool.integer(0_i32));
    let mut out = Vec::with_capacity(num as usize);
    for k in 0..num {
        if k > 0 && coeff_loop_should_stop(pool) {
            // Inside a `series` call this prefix is not an answer — record why,
            // for `series` to turn into a refusal. Every other caller wants the
            // prefix, so nothing is recorded for them and no stale refusal is
            // left behind for the next `take_series_refusal`.
            if IN_SERIES.with(|c| c.get()) {
                let refusal = SeriesRefusal {
                    requested: num,
                    computed: k,
                    budget: crate::budget::check().err(),
                };
                LAST_REFUSAL.with(|c| c.set(Some(refusal)));
            }
            break;
        }
        let ev = subs(cur, &mapping, pool);
        let simp = simplify(ev, pool).value;
        let fc = factorial_u32(k);
        let inv_fact = pool.rational(rug::Integer::from(1), fc);
        let coeff = simplify(pool.mul(vec![simp, inv_fact]), pool).value;
        out.push(coeff);
        if k + 1 < num {
            cur = diff(cur, xi, pool)?.value;
        }
    }
    Ok(out)
}

fn assemble_series(
    coeffs: &[ExprId],
    valuation: i32,
    h_expr: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Series {
    let mut terms = Vec::new();
    for (k, coeff) in coeffs.iter().enumerate() {
        if is_structural_zero(*coeff, pool) {
            continue;
        }
        let exp = valuation + k as i32;
        let pow_term = if exp == 0 {
            pool.integer(1_i32)
        } else if exp == 1 {
            h_expr
        } else {
            pool.pow(h_expr, pool.integer(exp as i64))
        };
        terms.push(pool.mul(vec![*coeff, pow_term]));
    }
    let big_o_pow = laurent_big_o_pow(valuation, order);
    let o_term = pool.big_o(pool.pow(h_expr, pool.integer(big_o_pow)));
    terms.push(o_term);
    Series(pool.add(terms))
}

fn expansion_matched_laurent(
    shifted: ExprId,
    xi: ExprId,
    h_expr: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<LocalExpansion, SeriesError> {
    let (nums, dens) = match collect_term_factors(shifted, pool) {
        Some(p) => p,
        None => {
            let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
            return Ok(LocalExpansion {
                valuation: 0,
                coeffs,
                h_expr,
            });
        }
    };

    let n_expr = product_sorted(pool, nums);
    let d_expr = product_sorted(pool, dens);

    let rf = match RationalFunction::from_symbolic(n_expr, d_expr, vec![xi], pool) {
        Ok(r) => r,
        Err(_) => {
            let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
            return Ok(LocalExpansion {
                valuation: 0,
                coeffs,
                h_expr,
            });
        }
    };

    if rf.numer.is_zero() {
        return Ok(LocalExpansion {
            valuation: 0,
            coeffs: vec![pool.integer(0_i32)],
            h_expr,
        });
    }

    let n_uni = match UniPoly::from_symbolic(rf.numer.to_expr(pool), xi, pool) {
        Ok(u) => u,
        Err(_) => {
            let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
            return Ok(LocalExpansion {
                valuation: 0,
                coeffs,
                h_expr,
            });
        }
    };
    let d_uni = match UniPoly::from_symbolic(rf.denom.to_expr(pool), xi, pool) {
        Ok(u) => u,
        Err(_) => {
            let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
            return Ok(LocalExpansion {
                valuation: 0,
                coeffs,
                h_expr,
            });
        }
    };

    let vn = match unipoly_valuation(&n_uni) {
        Some(v) => v,
        None => {
            return Ok(LocalExpansion {
                valuation: 0,
                coeffs: vec![pool.integer(0_i32)],
                h_expr,
            });
        }
    };
    let vd = match unipoly_valuation(&d_uni) {
        Some(v) => v,
        None => {
            let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
            return Ok(LocalExpansion {
                valuation: 0,
                coeffs,
                h_expr,
            });
        }
    };

    let valuation = vn as i32 - vd as i32;
    let n0 = unipoly_strip_low(&n_uni, vn);
    let d0 = unipoly_strip_low(&d_uni, vd);

    let d0c = d0.coefficients();
    if d0c.is_empty() || d0c[0] == 0 {
        let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
        return Ok(LocalExpansion {
            valuation: 0,
            coeffs,
            h_expr,
        });
    }

    let n0_e = n0.to_symbolic_expr(pool);
    let d0_e = d0.to_symbolic_expr(pool);
    let inv_d = pool.pow(d0_e, pool.integer(-1_i32));
    let g = simplify(pool.mul(vec![n0_e, inv_d]), pool).value;

    let num_taylor: u32 = if valuation < 0 {
        order
    } else {
        (order as i32 - valuation).max(0) as u32
    };

    if num_taylor == 0 {
        return Ok(LocalExpansion {
            valuation,
            coeffs: Vec::new(),
            h_expr,
        });
    }

    let coeffs = taylor_coefficients(g, xi, num_taylor, pool)?;
    Ok(LocalExpansion {
        valuation,
        coeffs,
        h_expr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprData};

    fn contains_big_o(id: ExprId, pool: &ExprPool) -> bool {
        match pool.get(id) {
            ExprData::BigO(_) => true,
            ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|e| contains_big_o(*e, pool)),
            ExprData::Pow { base, exp } => contains_big_o(base, pool) || contains_big_o(exp, pool),
            ExprData::Func { args, .. } => args.iter().any(|e| contains_big_o(*e, pool)),
            _ => false,
        }
    }

    #[test]
    fn series_cos_about_zero_has_big_o() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let z = p.integer(0);
        let cx = p.func("cos", vec![x]);
        let s = series(cx, x, z, 6, &p).unwrap();
        assert!(contains_big_o(s.expr(), &p));
    }

    #[test]
    fn series_inv_x_laurent_has_big_o() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let z = p.integer(0);
        let ix = p.pow(x, p.integer(-1));
        let s = series(ix, x, z, 4, &p).unwrap();
        assert!(contains_big_o(s.expr(), &p));
    }

    /// `√(t⁻² + t⁻¹)` at order 32 is the runaway shape: each coefficient is
    /// formed by differentiating the previous one without re-simplifying, and a
    /// nested radical's derivatives grow by a constant factor, so the loop is
    /// unfinishable rather than slow (order 13 already takes 0.15 s and the cost
    /// doubles per order).
    ///
    /// The refusal is the assertion. A *short* series would be worse than the
    /// hang it replaces: `O(t^32)` on nine computed coefficients is a false
    /// statement about the remainder, and unlike a timeout the caller has no way
    /// to notice. This test also passes trivially if the expansion is ever made
    /// to terminate honestly at the full order — see the `is_ok` arm.
    #[test]
    fn series_refuses_rather_than_truncating_a_runaway_radical() {
        use crate::errors::AlkahestError;
        let p = ExprPool::new();
        let t = p.symbol("t", Domain::Real);
        let inner = p.add(vec![p.pow(t, p.integer(-2)), p.pow(t, p.integer(-1))]);
        let ex = p.func("sqrt", vec![inner]);

        match series(ex, t, p.integer(0), 32, &p) {
            Ok(_) => {
                // A future fast path that really reaches order 32 is welcome;
                // it must not leave a refusal behind.
                assert_eq!(take_series_refusal(), None);
            }
            Err(e) => {
                assert!(matches!(e, SeriesError::InvalidOrder), "{e:?}");
                let refusal = take_series_refusal().expect("work-ceiling refusal recorded");
                assert_eq!(refusal.code(), "E-SERIES-003");
                assert_eq!(refusal.budget(), None, "no budget was active");
                assert!(
                    refusal.computed_coefficients() < refusal.requested_coefficients(),
                    "{refusal}"
                );
            }
        }
    }

    /// The carrier variant keeps its original meaning: `order == 0` is a user
    /// error, not a refusal, and must not leave a refusal pending for the
    /// bindings to mis-report as `E-SERIES-003`.
    #[test]
    fn order_zero_is_a_user_error_not_a_refusal() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let cx = p.func("cos", vec![x]);
        let err = series(cx, x, p.integer(0), 0, &p).unwrap_err();
        assert!(matches!(err, SeriesError::InvalidOrder), "{err:?}");
        assert_eq!(take_series_refusal(), None);
    }

    /// A budget trip is attributed to the budget, so a binding raises
    /// `E-BUDGET-*` rather than "this order is unreachable".
    #[test]
    fn budget_stops_a_series_and_is_attributed() {
        use crate::budget::{self, Budget, BudgetError};
        let p = ExprPool::new();
        let t = p.symbol("t", Domain::Real);
        let inner = p.add(vec![p.pow(t, p.integer(-2)), p.pow(t, p.integer(-1))]);
        let ex = p.func("sqrt", vec![inner]);

        let _guard = budget::enter(Budget::new().with_max_steps(3));
        let err = series(ex, t, p.integer(0), 32, &p).unwrap_err();
        assert!(matches!(err, SeriesError::InvalidOrder), "{err:?}");
        let refusal = take_series_refusal().expect("budget refusal recorded");
        assert!(
            matches!(refusal.budget(), Some(BudgetError::Steps { .. })),
            "{refusal}"
        );
    }

    /// The ceiling must not cost coverage: an ordinary high-order expansion of
    /// a function whose derivatives close still returns, and leaves no refusal.
    #[test]
    fn ordinary_high_order_expansion_is_unaffected() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let sx = p.func("sin", vec![x]);
        let s = series(sx, x, p.integer(0), 24, &p).unwrap();
        assert!(contains_big_o(s.expr(), &p));
        assert_eq!(take_series_refusal(), None);
    }
}
