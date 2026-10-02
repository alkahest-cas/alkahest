//! Truncated Taylor / Laurent series with symbolic [`crate::kernel::ExprData::BigO`] remainder (V2-15).

use crate::budget::BudgetError;
use crate::diff::{diff, DiffError};
use crate::flint::FlintPoly;
use crate::kernel::{subs, Domain, ExprData, ExprId, ExprPool};
use crate::poly::{together_parts, RationalFunction, UniPoly};
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
/// Let `h = var - point`. The returned expression is `∑ cₑ·hᵉ + O(h^order)`
/// over every exponent `valuation ≤ e < order`, for Taylor and Laurent series
/// alike (`1/x` about `0` at order 4 gives `x⁻¹ + O(x⁴)`) — the same meaning
/// `order` has for [`crate::calculus::puiseux::puiseux_series`].
///
/// For a pole this takes `order − valuation` Taylor coefficients of the
/// analytic factor `h^{-valuation} · f`. It used to take `order` of them and
/// label the result `O(h)`, which was false whenever the pole order was at
/// least `order`: `sin(x)⁻⁴` at order 4 lost its `11/45`.
///
/// # Method
///
/// When every coefficient is rational the expansion is computed with
/// truncated power-series arithmetic over ℚ (`calculus::tps`): each subterm
/// is expanded once and combined by convolution, Miller's power recurrence
/// and the ODE recurrences for the elementary heads. Everything else goes
/// through the route below — `order` rounds of differentiate, substitute,
/// simplify — and gets the same result where both apply.
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
///
/// # Singular expansion points
///
/// Substituting `point` into a derivative is not the same as taking a limit, so
/// a **removable** singularity — `sin(x)/x` at `0`, whose constant term is the
/// literal `0/0` — has no coefficient the direct route can form. Where the
/// expression can be put over a common denominator (`quotient_expansion`) the
/// two sides are expanded separately and the *power series* are divided, which
/// cancels the vanishing factor before anything is evaluated: `sin(x)/x` gives
/// `1 − x²/6 + O(x⁴)`, `1/sin(x)` gives `x⁻¹ + x/6 + O(x)`.
///
/// Where it cannot — a branch point (`√x`), a logarithmic singularity
/// (`log x`), an essential one (`e^{1/x}`) — there is no Laurent expansion to
/// return, and this is an **`Err`** carrying a
/// [`SeriesRefusalCause::IndeterminateCoefficient`] refusal (`E-SERIES-004`).
/// It is never a `Series` whose coefficients contain `0⁻¹`: that reports
/// success, cannot be evaluated, and evaluates to `NaN` for whoever tries.
///
/// A **branch point** is the one of those three that has an expansion this
/// function cannot hold: `√x = x^{1/2}` is a perfectly good *Puiseux* series,
/// and [`crate::calculus::puiseux::puiseux_series`] returns it. `Series` is a
/// bare `ExprId` with no ramification index to report, and every consumer of
/// `local_expansion` — `limit`, `gruntz`, `asymptotic`, `fps` — reads an
/// *integer* valuation, so the fractional case lives in a sibling type rather
/// than widening this one. The refusal here is unchanged and deliberate.
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

    let mut exp = local_expansion(expr, var, point, order, pool)?;

    if frame.refusal_pending() {
        return Err(SeriesError::InvalidOrder);
    }

    // `local_expansion` hands back `order` coefficients *of the unit part*,
    // i.e. exponents `valuation .. valuation + order`. For a pole that stops
    // short of `order` by exactly the pole order: `sin(x)⁻⁴` at order 4 gave
    // `x⁻⁴ + ⅔x⁻² + O(x)`, silently dropping the `11/45` that `O(x)` claims
    // is there. The working precision has to be raised by the valuation
    // shift, so re-expand with `order − valuation` unit-part coefficients.
    // A pole whose order was only discovered by the first pass may be deeper
    // on the second (the quotient route probes `order + slack` coefficients
    // for a valuation), so retry a bounded number of times and refuse rather
    // than label a short series `O(h^order)`.
    let mut attempts = 0;
    while exp.valuation < 0
        && i64::from(exp.valuation) + (exp.coeffs.len() as i64) < i64::from(order)
    {
        attempts += 1;
        let widened = order.checked_add(exp.valuation.unsigned_abs());
        let (Some(widened), true) = (widened, attempts <= 3) else {
            LAST_REFUSAL.with(|c| {
                c.set(Some(SeriesRefusal {
                    requested: order,
                    computed: exp.coeffs.len() as u32,
                    budget: None,
                    cause: SeriesRefusalCause::Exhausted,
                }))
            });
            return Err(SeriesError::InvalidOrder);
        };
        exp = local_expansion(expr, var, point, widened, pool)?;
        if frame.refusal_pending() {
            return Err(SeriesError::InvalidOrder);
        }
    }
    let LocalExpansion {
        valuation,
        coeffs,
        h_expr,
    } = exp;

    // A coefficient that is `0/0`, `1/0` or `log(0)` is not a value, and a
    // `Series` carrying one is the exact failure mode this module exists to
    // avoid: `Ok`, unevaluable, unsimplifiable, `NaN` on contact. Refuse.
    if let Some(idx) = first_indeterminate(&coeffs, pool) {
        LAST_REFUSAL.with(|c| {
            c.set(Some(SeriesRefusal {
                requested: coeffs.len() as u32,
                computed: idx as u32,
                budget: None,
                cause: SeriesRefusalCause::IndeterminateCoefficient,
            }))
        });
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

    // Fast path: when every coefficient is rational, expand compositionally
    // with truncated power-series arithmetic instead of `order` rounds of
    // differentiate-substitute-simplify. Same coefficients, same literals;
    // anything it does not cover falls through to the route below unchanged.
    if tps_enabled() {
        if let Some((valuation, coeffs)) = super::tps::tps_expansion(shifted, xi, order, pool) {
            let coeffs = coeffs
                .iter()
                .map(|r| super::tps::rational_literal(r, pool))
                .collect();
            return Ok(LocalExpansion {
                valuation,
                coeffs,
                h_expr,
            });
        }
    }

    let direct = expansion_matched_laurent(shifted, xi, h_expr, order, pool)?;
    if first_indeterminate(&direct.coeffs, pool).is_none() {
        return Ok(direct);
    }

    // A coefficient is `0/0`, `1/0`, `log(0)`, … — the expansion point is
    // singular for the *form* the coefficients were computed in, which is not
    // the same thing as singular for the function. Try the quotient route,
    // which divides power series instead of substituting into a quotient and
    // therefore sees through a removable singularity.
    //
    // The refusal channel is saved and restored so a probe that runs out of
    // room cannot make a *successful* repair look like a work-ceiling trip.
    let saved = LAST_REFUSAL.with(|c| c.get());
    if let Some(repaired) = quotient_expansion(shifted, xi, h_expr, order, pool)? {
        if first_indeterminate(&repaired.coeffs, pool).is_none() {
            LAST_REFUSAL.with(|c| c.set(saved));
            return Ok(repaired);
        }
    }

    // A removable singularity *inside* a function argument — `log(x·x⁻¹)`,
    // `exp(x·(−2x⁻¹)⁻¹)` — is invisible to the quotient route, which treats
    // `log(…)` as an opaque generator. Simplifying the shifted expression
    // cancels it (`x·x⁻¹ ↦ 1`), which is sound here because a series is a
    // statement about a punctured neighbourhood: the two forms agree there.
    // Tried last so it changes nothing that already expanded.
    let simplified = simplify(shifted, pool).value;
    if simplified != shifted {
        let again = expansion_matched_laurent(simplified, xi, h_expr, order, pool)?;
        if first_indeterminate(&again.coeffs, pool).is_none() {
            LAST_REFUSAL.with(|c| c.set(saved));
            return Ok(again);
        }
        if let Some(repaired) = quotient_expansion(simplified, xi, h_expr, order, pool)? {
            if first_indeterminate(&repaired.coeffs, pool).is_none() {
                LAST_REFUSAL.with(|c| c.set(saved));
                return Ok(repaired);
            }
        }
    }

    // The repair does not apply (branch point, essential singularity, a
    // denominator whose valuation is not visible). Hand back the direct
    // expansion unchanged: every caller other than `series` — `limit`,
    // `gruntz`, `asymptotic`, `fps` — already inspects coefficients for
    // usability and has its own fallback, and `series` itself turns an
    // indeterminate coefficient into a refusal rather than a value.
    Ok(direct)
}

#[cfg(test)]
thread_local! {
    /// Test-only switch back to the general route, so the differential tests
    /// can compare the fast path against it in one pool.
    static TPS_DISABLED: Cell<bool> = const { Cell::new(false) };
}

/// Run `f` with the truncated-power-series fast path switched off.
#[cfg(test)]
pub(crate) fn without_tps<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            TPS_DISABLED.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(TPS_DISABLED.with(|c| c.replace(true)));
    f()
}

fn tps_enabled() -> bool {
    #[cfg(test)]
    {
        !TPS_DISABLED.with(|c| c.get())
    }
    #[cfg(not(test))]
    {
        true
    }
}

/// Expand `shifted` about `ξ = 0` by **dividing power series** rather than
/// substituting into a quotient.
///
/// This is the removable-singularity repair. `sin(ξ)/ξ` has no Taylor
/// coefficient at `ξ = 0` when you substitute — the constant term is the
/// literal `0/0` — but it has a perfectly ordinary one when you write it as
/// `(ξ − ξ³/6 + …)/(ξ)` and divide the two expansions, because the vanishing
/// factor cancels *before* anything is evaluated at `0`.
///
/// The steps:
///
/// 1. [`together_parts`] puts the expression over one denominator, treating
///    `sin(ξ)`, `log(1+ξ)`, `ξ^n` and friends as opaque generators. This is
///    what turns `1/ξ − 1/sin ξ` (an `Add` of two poles that cancel, which
///    `collect_term_factors` cannot even take apart) into the single quotient
///    `(sin ξ − ξ)/(ξ · sin ξ)`.
/// 2. Taylor-expand numerator and denominator separately. Both are polynomials
///    in the generators, hence analytic wherever the generators are, so their
///    coefficients are ordinary substitutions — no indeterminate forms.
/// 3. Read off the valuations `vₙ`, `v_d` (index of the first coefficient that
///    is not the literal `0`) and divide the unit parts by the standard
///    recurrence `cₖ = (aₖ − Σ_{j≥1} b_j c_{k−j}) / b₀`.
///
/// Returns `Ok(None)` when the shape does not apply: no denominator, a
/// denominator that does not vanish at the point (the direct route was already
/// right, so a failure there is not this function's to fix), a valuation that
/// is not visible within the probe depth, or a numerator/denominator that is
/// itself singular at `0` (`log(ξ)`, `√ξ` — genuine branch points, for which
/// no Laurent expansion exists and a refusal is the correct answer).
///
/// # Zero tests are one-sided, on purpose
///
/// A valuation is found by skipping coefficients that are the *structural*
/// integer `0`. A coefficient that is secretly zero but does not look it stops
/// the scan early, which **understates** the valuation. That is the safe
/// direction for the numerator: `c₀` comes out `0`, the leading zeros are
/// dropped by [`assemble_series`], and the series is still correct (merely
/// computed to a lower valuation than necessary). For the denominator it means
/// dividing by a `b₀` that is really `0`, which produces `0⁻¹` — caught by
/// [`coefficient_is_indeterminate`] and refused. Neither direction can return
/// a wrong series.
fn quotient_expansion(
    shifted: ExprId,
    xi: ExprId,
    h_expr: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<Option<LocalExpansion>, SeriesError> {
    let Ok((num, den)) = together_parts(shifted, vec![xi], pool) else {
        return Ok(None);
    };
    if num == shifted {
        // Nothing was combined or cancelled — re-expanding would reproduce the
        // same indeterminate coefficients. This is the branch-point /
        // essential-singularity exit (`√ξ`, `log ξ`, `e^{1/ξ}`).
        return Ok(None);
    }
    if matches!(pool.get(den), ExprData::Integer(n) if n.0 == 1) {
        // The GCD reduction removed the vanishing factor outright — the
        // `(x²−1)/(x−1)` shape, where the quotient *is* a polynomial and there
        // is nothing left to divide. Expanding the cancelled form is the whole
        // repair.
        let coeffs = taylor_coefficients(num, xi, order, pool)?;
        if coeffs.len() < order as usize {
            return Ok(None);
        }
        return Ok(Some(LocalExpansion {
            valuation: 0,
            coeffs,
            h_expr,
        }));
    }

    let Some((d_probe, vd)) = probe_valuation(den, xi, order, pool)? else {
        return Ok(None);
    };
    // `vd == 0` is not an exit: the denominator is a unit at the point, and
    // the direct route can still have failed because the *form* was singular
    // there (`((x−2)/x)⁻¹`, which `together` turns into `x/(x−2)`). Dividing
    // two analytic series by a unit is exactly right; a numerator that is
    // itself singular is caught by the indeterminate check on `a` below.
    let Some((n_probe, vn)) = probe_valuation(num, xi, order, pool)? else {
        return Ok(None);
    };

    let valuation = vn as i32 - vd as i32;
    let num_taylor: u32 = if valuation < 0 {
        order
    } else {
        (order as i32 - valuation).max(0) as u32
    };
    if num_taylor == 0 {
        return Ok(Some(LocalExpansion {
            valuation,
            coeffs: Vec::new(),
            h_expr,
        }));
    }

    // Unit parts: `a_i = numerator coefficient (vn + i)`, likewise for `b`.
    let a = unit_part(&n_probe, num, xi, vn, num_taylor, pool)?;
    let b = unit_part(&d_probe, den, xi, vd, num_taylor, pool)?;
    let (Some(a), Some(b)) = (a, b) else {
        return Ok(None);
    };
    if a.iter()
        .chain(b.iter())
        .any(|&c| coefficient_is_indeterminate(c, pool))
    {
        // A generator that is not analytic at the point (`log ξ`, `√ξ`).
        return Ok(None);
    }

    let inv_b0 = reciprocal(b[0], pool);
    let mut coeffs: Vec<ExprId> = Vec::with_capacity(num_taylor as usize);
    for k in 0..num_taylor as usize {
        let mut terms = vec![a[k]];
        for (j, &bj) in b.iter().enumerate().take(k + 1).skip(1) {
            terms.push(pool.mul(vec![pool.integer(-1_i32), bj, coeffs[k - j]]));
        }
        let numer = if terms.len() == 1 {
            terms[0]
        } else {
            pool.add(terms)
        };
        coeffs.push(simplify(pool.mul(vec![numer, inv_b0]), pool).value);
        if coeff_loop_should_stop(pool) {
            return Ok(None);
        }
    }

    Ok(Some(LocalExpansion {
        valuation,
        coeffs,
        h_expr,
    }))
}

/// How many coefficients past the requested order [`quotient_expansion`]
/// first looks at while hunting for a valuation.
///
/// The numerator and denominator valuations are pole/zero orders, which are
/// small in most expansions (`sin(x)/x`: 1 and 1; `(1−cos x)/x²`: 2 and 2),
/// so the first probe is cheap. A deeper zero — `(1−cos x)³` has valuation 6,
/// which a first probe at order 1 does not reach — gets a wider probe, up to
/// [`VALUATION_PROBE_MAX_SLACK`]; every probe runs under the coefficient-loop
/// ceiling and the ambient budget, so a failed one cannot run away.
const VALUATION_PROBE_SLACK: usize = 4;

/// The widest valuation probe, past the requested order.
const VALUATION_PROBE_MAX_SLACK: usize = 64;

/// Taylor coefficients of `expr` at `ξ = 0` deep enough to see its valuation,
/// and that valuation — or `None` when no nonzero coefficient appears within
/// `order + VALUATION_PROBE_MAX_SLACK` (or the coefficient loop was stopped).
///
/// Before the probe widened, a pole deeper than `order + 3` was refused as
/// "indeterminate": `sin(x)⁻⁵` expanded at order 5 but not at order 1.
fn probe_valuation(
    expr: ExprId,
    xi: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<Option<(Vec<ExprId>, usize)>, SeriesError> {
    let mut slack = VALUATION_PROBE_SLACK;
    loop {
        let probe = (order as usize).saturating_add(slack);
        let probe_u32 = probe.min(u32::MAX as usize) as u32;
        let coeffs = taylor_coefficients(expr, xi, probe_u32, pool)?;
        if let Some(v) = leading_index(&coeffs, pool) {
            return Ok(Some((coeffs, v)));
        }
        if coeffs.len() < probe_u32 as usize || slack >= VALUATION_PROBE_MAX_SLACK {
            // Stopped by the ceiling/budget, or genuinely deeper than we look.
            return Ok(None);
        }
        slack = (slack * 4).min(VALUATION_PROBE_MAX_SLACK);
    }
}

/// `1/expr`, folded when `expr` is a nonzero rational literal.
///
/// `simplify` leaves `(-1/2)^-1` alone, which is correct but reads badly in
/// every coefficient of the result (`1/(1−cos x)` would print
/// `−(−1/2)⁻¹·x⁻² + …` instead of `2·x⁻² + …`). The leading coefficient of a
/// divisor is a literal in essentially every expansion, so folding it here is
/// worth the four lines.
fn reciprocal(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Integer(n) if n.0 != 0 => pool.rational(rug::Integer::from(1), n.0.clone()),
        ExprData::Rational(r) if r.0 != 0 => {
            let inv = r.0.clone().recip();
            let (num, den) = inv.into_numer_denom();
            pool.rational(num, den)
        }
        _ => simplify(pool.pow(expr, pool.integer(-1_i32)), pool).value,
    }
}

/// Index of the first coefficient that is not the literal integer `0`.
///
/// `None` when every probed coefficient is zero — either the expression really
/// is identically zero to this depth, or the valuation is deeper than the
/// probe. The two are indistinguishable from here, so the caller declines
/// rather than assuming either.
fn leading_index(coeffs: &[ExprId], pool: &ExprPool) -> Option<usize> {
    coeffs.iter().position(|&c| !is_structural_zero(c, pool))
}

/// Coefficients `v, v+1, …, v+wanted-1` of `expr`, re-expanding when the probe
/// did not reach that far.
fn unit_part(
    probed: &[ExprId],
    expr: ExprId,
    xi: ExprId,
    v: usize,
    wanted: u32,
    pool: &ExprPool,
) -> Result<Option<Vec<ExprId>>, SeriesError> {
    let need = v.saturating_add(wanted as usize);
    if probed.len() >= need {
        return Ok(Some(probed[v..need].to_vec()));
    }
    let Ok(need_u32) = u32::try_from(need) else {
        return Ok(None);
    };
    let full = taylor_coefficients(expr, xi, need_u32, pool)?;
    if full.len() < need {
        // The coefficient loop stopped early (ceiling or budget).
        return Ok(None);
    }
    Ok(Some(full[v..need].to_vec()))
}

fn factorial_u32(n: u32) -> rug::Integer {
    let mut r = rug::Integer::from(1);
    for i in 2..=n {
        r *= i;
    }
    r
}

pub(crate) fn expansion_increment(pool: &ExprPool, var: ExprId, point: ExprId) -> ExprId {
    match pool.get(point) {
        ExprData::Integer(n) if n.0 == 0 => var,
        _ => pool.add(vec![var, pool.mul(vec![pool.integer(-1_i32), point])]),
    }
}

pub(crate) fn is_structural_zero(id: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(id), ExprData::Integer(n) if n.0 == 0)
}

// ---------------------------------------------------------------------------
// Indeterminate coefficients
// ---------------------------------------------------------------------------

/// True when `expr` contains a `0^n` node with `n` a negative constant.
///
/// Subsumed in the engine by [`has_singular_node`] (whose syntactic half this
/// is, and the only half that can see through a free parameter: `k · 0⁻¹` is
/// not a number however unknown `k` is); kept as the tests' independent
/// oracle for "no successful series carries a `0⁻ⁿ`".
///
/// The twin of [`crate::calculus::limits`]'s check of the same name; the two
/// are deliberately separate because their verdicts differ on `±∞` (an
/// established divergence is a fine *limit* and never a *series coefficient*).
#[cfg(test)]
fn contains_zero_to_negative_power(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let zero_base = matches!(pool.get(base), ExprData::Integer(n) if n.0 == 0);
            let negative_exp = match pool.get(exp) {
                ExprData::Integer(n) => n.0 < 0,
                ExprData::Rational(r) => r.0 < 0,
                ExprData::Float(f) => f.inner.to_f64() < 0.0,
                _ => false,
            };
            (zero_base && negative_exp)
                || contains_zero_to_negative_power(base, pool)
                || contains_zero_to_negative_power(exp, pool)
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            xs.iter().any(|&x| contains_zero_to_negative_power(x, pool))
        }
        ExprData::Func { args, .. } => args
            .iter()
            .any(|&a| contains_zero_to_negative_power(a, pool)),
        _ => false,
    }
}

/// True when this Taylor/Laurent coefficient is provably **not a number**.
///
/// Coefficients are formed by substituting the expansion point into repeated
/// derivatives (see [`taylor_coefficients`]), so at a singular point the
/// substitution produces the indeterminate form itself — `0·0⁻¹` for
/// `sin(x)/x`, `sqrt(0)⁻¹` for `√x`, `log(0)` for `log x`, `exp(0⁻¹)` for
/// `e^{1/x}`. Every one of those is `NaN` or `±∞` when evaluated, and a
/// `Series` carrying one is worse than no answer: it reports success, cannot be
/// evaluated, and cannot be simplified.
///
/// Only **positive evidence** counts, in both directions:
///
/// * a `0^{negative}` node anywhere, which survives free parameters;
/// * any *sub*-expression the interpreter actually evaluated, to a non-finite
///   number.
///
/// The search has to go inside the coefficient rather than testing it whole,
/// because a free parameter makes the whole thing unevaluable while leaving
/// the culprit in plain sight: the `x¹` coefficient of `k·√x` is
/// `k · ½ · sqrt(0)⁻¹`, which `eval_interp` declines (it cannot value `k`)
/// even though `sqrt(0)⁻¹` on its own is `+∞`. Testing only the top node let
/// that through.
///
/// A coefficient the interpreter cannot reach a verdict on anywhere — one that
/// is just `cos(a)` for a symbolic expansion point, or one whose head has no
/// interpreter entry — is *not* indeterminate. That direction matters: this
/// predicate gates a refusal, so a false positive turns a working expansion
/// into an error.
fn coefficient_is_indeterminate(expr: ExprId, pool: &ExprPool) -> bool {
    has_singular_node(expr, pool) || has_non_finite_constant(expr, pool)
}

/// True when some subtree is closed arithmetic that evaluates to `NaN` or `±∞`.
///
/// `eval_interp` returns `Some` exactly for a subtree with no free symbols, so
/// a `Some` verdict is final for that subtree and the walk stops there;
/// recursion only follows branches that still mention a symbol, and those are
/// the only ones that can hide a constant the top-level call could not see.
fn has_non_finite_constant(expr: ExprId, pool: &ExprPool) -> bool {
    let env = HashMap::new();
    if let Some(v) = crate::jit::eval_interp(expr, &env, pool) {
        return !v.is_finite();
    }
    match pool.get(expr) {
        ExprData::Add(xs) | ExprData::Mul(xs) | ExprData::Func { args: xs, .. } => {
            xs.iter().any(|&c| has_non_finite_constant(c, pool))
        }
        ExprData::Pow { base, exp } => {
            has_non_finite_constant(base, pool) || has_non_finite_constant(exp, pool)
        }
        _ => false,
    }
}

/// True when some subtree divides by, or takes the `log` of, a constant that
/// is `0`: the syntactic `0^{negative}` (which survives a free parameter),
/// and also the closed forms the interpreter values at exactly `0` —
/// `asin(0)⁻¹`, `(0 + 0)⁻¹`, `sqrt(0)⁻¹`, `log(0)`.
///
/// [`has_non_finite_constant`] stops at the first subtree the interpreter can
/// value, and IEEE arithmetic can turn the singularity into a finite number on
/// the way up: `2·log(0)⁻¹` is `−0.0`, `(−2·asin(0)⁻¹)⁻¹` is `−0.0`. Both were
/// accepted as coefficients (`x²/log x` returned a series). This looks at the
/// singular nodes themselves. Only a negative-power or `log` node costs an
/// evaluation, and the walk allocates nothing, so an ordinary coefficient
/// (`sin`/`cos` of `0`) costs one traversal — this runs on every raw Taylor
/// coefficient.
fn has_singular_node(expr: ExprId, pool: &ExprPool) -> bool {
    fn is_zero_constant(e: ExprId, pool: &ExprPool) -> bool {
        if matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0) {
            return true;
        }
        crate::jit::eval_interp(e, &HashMap::new(), pool) == Some(0.0)
    }
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let negative = match pool.get(exp) {
                ExprData::Integer(n) => n.0 < 0,
                ExprData::Rational(r) => r.0 < 0,
                ExprData::Float(f) => f.inner.to_f64() < 0.0,
                _ => false,
            };
            (negative && is_zero_constant(base, pool))
                || has_singular_node(base, pool)
                || has_singular_node(exp, pool)
        }
        ExprData::Func { name, args } => {
            (name == "log" && args.len() == 1 && is_zero_constant(args[0], pool))
                || args.iter().any(|&a| has_singular_node(a, pool))
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&x| has_singular_node(x, pool)),
        _ => false,
    }
}

/// Index of the first coefficient that is not a number, if any.
pub(crate) fn first_indeterminate(coeffs: &[ExprId], pool: &ExprPool) -> Option<usize> {
    coeffs
        .iter()
        .position(|&c| coefficient_is_indeterminate(c, pool))
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
    cause: SeriesRefusalCause,
}

/// Why a [`SeriesRefusal`] was raised.
///
/// Non-exhaustive so a future refusal reason is an additive change; matching
/// callers must carry a `_` arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SeriesRefusalCause {
    /// The coefficient loop ran past [`MAX_SERIES_POOL_GROWTH`] (or an active
    /// [`crate::budget`]) before reaching the requested order — `E-SERIES-003`.
    Exhausted,
    /// A coefficient came out as an *indeterminate form* rather than a value:
    /// `0·0⁻¹`, `0⁻¹`, `log(0)`, anything that evaluates to `NaN` or `±∞`
    /// — `E-SERIES-004`.
    ///
    /// This is what a **removable singularity** at the expansion point looks
    /// like from inside the coefficient loop. Coefficients are formed by
    /// substituting the expansion point into repeated derivatives, so
    /// `sin(x)/x` at `0` yields the literal quotient `0/0` for its constant
    /// term where the *limit* is `1`. `local_expansion` repairs the common
    /// shape (see `quotient_expansion`); when the repair does not apply, the
    /// coefficient is not a number and the expansion is refused rather than
    /// returned with a `NaN` inside it.
    IndeterminateCoefficient,
}

impl SeriesRefusal {
    /// Why the expansion was refused.
    pub fn cause(&self) -> SeriesRefusalCause {
        self.cause
    }
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
        match self.cause {
            SeriesRefusalCause::Exhausted => write!(
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
            ),
            SeriesRefusalCause::IndeterminateCoefficient => write!(
                f,
                "coefficient {} of {} is an indeterminate form (0/0, 1/0, log(0) or \
                 similar), not a number: the expansion point is a singularity this \
                 expansion cannot resolve. Refusing to return a series whose \
                 coefficients evaluate to NaN or infinity",
                self.computed, self.requested,
            ),
        }
    }
}

impl std::error::Error for SeriesRefusal {}

impl crate::errors::AlkahestError for SeriesRefusal {
    fn code(&self) -> &'static str {
        match self.cause {
            SeriesRefusalCause::Exhausted => "E-SERIES-003",
            SeriesRefusalCause::IndeterminateCoefficient => "E-SERIES-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self.cause {
            SeriesRefusalCause::Exhausted => Some(
                "ask for a lower order, raise the budget, or rewrite the expression so its \
                 repeated derivatives close (nested radicals grow by a constant factor per \
                 coefficient)",
            ),
            SeriesRefusalCause::IndeterminateCoefficient => Some(
                "the expansion point is a branch point, an essential singularity, or a \
                 removable one this engine cannot cancel: cancel the singular factor by \
                 hand (`cancel`/`together`), expand about a nearby regular point, or use \
                 `limit` for the single value you need",
            ),
        }
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
                    cause: SeriesRefusalCause::Exhausted,
                };
                LAST_REFUSAL.with(|c| c.set(Some(refusal)));
            }
            break;
        }
        let ev = subs(cur, &mapping, pool);
        // Syntactic and singular-node checks only: a whole-tree numeric
        // evaluation here doubled the cost of every ordinary expansion, and
        // the forms `simplify` launders are exactly a negative power or `log`
        // of a closed zero, which these see without evaluating anything else.
        if has_singular_node(ev, pool) {
            // The derivative is singular *in form* at the point. `simplify`
            // is free to launder that into a number — `(−2·0⁻¹)⁻¹ ↦ 0` by the
            // power rule `(ab)⁻¹ = a⁻¹b⁻¹`, `0·(1 + 0⁻¹) ↦ 0`,
            // `0·sqrt(0)⁻¹ ↦ 0` — and did: `x/(x−2)` written as `((x−2)/x)⁻¹`
            // expanded to a bare `O(x⁴)`, `x·(1/x + 1)` lost its constant
            // term, `x²·√(x³)` came back as `0 + O(x⁴)`. Keep the raw
            // substitution so the coefficient is recognised as indeterminate
            // and the caller takes the quotient route (or refuses).
            out.push(ev);
        } else {
            let simp = simplify(ev, pool).value;
            let fc = factorial_u32(k);
            let inv_fact = pool.rational(rug::Integer::from(1), fc);
            let coeff = simplify(pool.mul(vec![simp, inv_fact]), pool).value;
            out.push(coeff);
        }
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
        if i64::from(exp) >= i64::from(order) {
            // Computed as a by-product of the pole shift; past the requested
            // truncation, so it belongs to the remainder.
            break;
        }
        let pow_term = if exp == 0 {
            pool.integer(1_i32)
        } else if exp == 1 {
            h_expr
        } else {
            pool.pow(h_expr, pool.integer(exp as i64))
        };
        terms.push(pool.mul(vec![*coeff, pow_term]));
    }
    // Every term with exponent `< order` is present (see `series`), so the
    // remainder is `O(h^order)` for Taylor and Laurent series alike. It used
    // to be `O(h)` for every Laurent series, which is false whenever the pole
    // order is at least `order`.
    let o_term = pool.big_o(pool.pow(h_expr, pool.integer(i64::from(order))));
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

    // -----------------------------------------------------------------------
    // Truncation at a negative valuation (audit A9)
    // -----------------------------------------------------------------------

    /// `(exponent, numeric coefficient)` pairs and the `O(x^k)` exponent of a
    /// `series` result.
    fn series_terms(s: Series, x: ExprId, p: &ExprPool) -> (Vec<(i64, f64)>, i64) {
        let power_of_x = |e: ExprId| -> Option<i64> {
            if e == x {
                return Some(1);
            }
            match p.get(e) {
                ExprData::Pow { base, exp } if base == x => match p.get(exp) {
                    ExprData::Integer(n) => n.0.to_i64(),
                    _ => None,
                },
                _ => None,
            }
        };
        let parts = match p.get(s.expr()) {
            ExprData::Add(xs) => xs,
            _ => vec![s.expr()],
        };
        let env = HashMap::new();
        let mut terms = Vec::new();
        let mut big_o = None;
        for t in parts {
            if let ExprData::BigO(inner) = p.get(t) {
                big_o = power_of_x(inner);
                continue;
            }
            let factors = match p.get(t) {
                ExprData::Mul(xs) => xs,
                _ => vec![t],
            };
            let mut k = 0;
            let mut c = 1.0;
            for f in factors {
                match power_of_x(f) {
                    Some(e) => k += e,
                    None => c *= crate::jit::eval_interp(f, &env, p).expect("numeric coefficient"),
                }
            }
            terms.push((k, c));
        }
        terms.sort_by_key(|&(k, _)| k);
        (terms, big_o.expect("an O(.) term"))
    }

    fn assert_terms(got: &[(i64, f64)], want: &[(i64, f64)]) {
        assert_eq!(got.len(), want.len(), "got {got:?} want {want:?}");
        for ((ge, gc), (we, wc)) in got.iter().zip(want) {
            assert!(
                ge == we && (gc - wc).abs() < 1e-12,
                "got {got:?} want {want:?}"
            );
        }
    }

    /// `sin(x)⁻⁴` at order 4 printed `x⁻⁴ + ⅔x⁻² + O(x)`, which claims the
    /// `x⁰` coefficient is present. It is `11/45` (Mathematica `Series`).
    #[test]
    fn a_pole_deeper_than_the_order_keeps_every_term_below_it() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pow(p.func("sin", vec![x]), p.integer(-4));
        let (terms, big_o) = series_terms(series(f, x, p.integer(0), 4, &p).unwrap(), x, &p);
        assert_terms(
            &terms,
            &[
                (-4, 1.0),
                (-2, 2.0 / 3.0),
                (0, 11.0 / 45.0),
                (2, 62.0 / 945.0),
            ],
        );
        assert_eq!(big_o, 4);
    }

    /// `(x + x²)⁻³ = x⁻³(1+x)⁻³`; order 3 must reach `x²`.
    #[test]
    fn a_rational_pole_is_expanded_to_the_requested_order() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pow(p.add(vec![x, p.pow(x, p.integer(2))]), p.integer(-3));
        let (terms, big_o) = series_terms(series(f, x, p.integer(0), 3, &p).unwrap(), x, &p);
        assert_terms(
            &terms,
            &[
                (-3, 1.0),
                (-2, -3.0),
                (-1, 6.0),
                (0, -10.0),
                (1, 15.0),
                (2, -21.0),
            ],
        );
        assert_eq!(big_o, 3);
    }

    /// `((x−2)/x)⁻¹ = x/(x−2)` expanded to a bare `O(x⁴)`: every coefficient
    /// was `(−2·0⁻¹)^{-k}`-shaped and `simplify` folded it to `0`.
    #[test]
    fn a_singular_form_of_an_analytic_function_is_not_folded_to_zero() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let q = p.mul(vec![p.add(vec![x, p.integer(-2)]), p.pow(x, p.integer(-1))]);
        let f = p.pow(q, p.integer(-1));
        let (terms, big_o) = series_terms(series(f, x, p.integer(0), 4, &p).unwrap(), x, &p);
        assert_terms(&terms, &[(1, -0.5), (2, -0.25), (3, -0.125)]);
        assert_eq!(big_o, 4);
    }

    #[test]
    fn a_simple_pole_carries_the_requested_remainder() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pow(x, p.integer(-1));
        let (terms, big_o) = series_terms(series(f, x, p.integer(0), 4, &p).unwrap(), x, &p);
        assert_terms(&terms, &[(-1, 1.0)]);
        assert_eq!(big_o, 4);
    }

    // -----------------------------------------------------------------------
    // Removable singularities at the expansion point
    // -----------------------------------------------------------------------

    /// Numeric value of every coefficient, in order, or `None` where the
    /// coefficient is not a closed constant.
    fn coeff_values(
        expr: ExprId,
        var: ExprId,
        point: ExprId,
        order: u32,
        p: &ExprPool,
    ) -> Vec<f64> {
        let exp = local_expansion(expr, var, point, order, p).expect("expansion");
        let env = HashMap::new();
        exp.coeffs
            .iter()
            .map(|&c| crate::jit::eval_interp(c, &env, p).unwrap_or(f64::NAN))
            .collect()
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "got {got:?} want {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "got {got:?} want {want:?}");
        }
    }

    fn sin_over_x(p: &ExprPool) -> ExprId {
        let x = p.symbol("x", Domain::Real);
        let s = p.func("sin", vec![x]);
        p.mul(vec![s, p.pow(x, p.integer(-1))])
    }

    /// The headline case. `sin(x)/x` has a *removable* singularity at `0`: the
    /// function extends analytically with value `1`, and SymPy, Maxima and
    /// every textbook give `1 − x²/6 + x⁴/120 + O(x⁶)`.
    ///
    /// Forming coefficients by substituting `0` into repeated derivatives
    /// cannot see that — the constant term comes out as the literal `0·0⁻¹` —
    /// and the result used to be returned as a *successful* `Series` full of
    /// `1/0` and `0/0`, which no caller could evaluate and none could detect.
    /// This is the regression test for the most common expansion in applied
    /// mathematics.
    #[test]
    fn sin_over_x_expands_through_its_removable_singularity() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let e = sin_over_x(&p);
        assert_close(
            &coeff_values(e, x, p.integer(0), 6, &p),
            &[1.0, 0.0, -1.0 / 6.0, 0.0, 1.0 / 120.0, 0.0],
        );
        assert!(series(e, x, p.integer(0), 6, &p).is_ok());
        assert_eq!(take_series_refusal(), None);
    }

    #[test]
    fn tan_over_x_expands_through_its_removable_singularity() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let t = p.func("tan", vec![x]);
        let e = p.mul(vec![t, p.pow(x, p.integer(-1))]);
        assert_close(
            &coeff_values(e, x, p.integer(0), 5, &p),
            &[1.0, 0.0, 1.0 / 3.0, 0.0, 2.0 / 15.0],
        );
    }

    /// `(1 − cos x)/x²` — a double zero over a double zero, so the repair has
    /// to line up two valuations rather than one.
    #[test]
    fn matched_double_zeros_cancel() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let c = p.func("cos", vec![x]);
        let n = p.add(vec![p.integer(1), p.mul(vec![p.integer(-1), c])]);
        let e = p.mul(vec![n, p.pow(x, p.integer(-2))]);
        assert_close(
            &coeff_values(e, x, p.integer(0), 5, &p),
            &[0.5, 0.0, -1.0 / 24.0, 0.0, 1.0 / 720.0],
        );
    }

    /// `1/x − 1/sin x` is a *sum* of two poles that cancel, which the
    /// factor-splitting fast path cannot even take apart. Putting it over a
    /// common denominator first is what makes it expandable.
    #[test]
    fn a_sum_of_cancelling_poles_expands() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let s = p.func("sin", vec![x]);
        let e = p.add(vec![
            p.pow(x, p.integer(-1)),
            p.mul(vec![p.integer(-1), p.pow(s, p.integer(-1))]),
        ]);
        // 1/x − 1/sin x = −x/6 − 7x³/360 + O(x⁵): valuation 1. The general
        // route reports that as `valuation: 1` with the coefficients of
        // x¹…x⁴; the power-series fast path as `valuation: 0` with a leading
        // zero, as the direct Taylor route does. Both cover x⁰…x⁴.
        let exp = local_expansion(e, x, p.integer(0), 5, &p).unwrap();
        let lead = exp
            .coeffs
            .iter()
            .position(|&c| !is_structural_zero(c, &p))
            .unwrap();
        assert_eq!(exp.valuation + lead as i32, 1);
        assert_eq!(exp.valuation + exp.coeffs.len() as i32, 5);
        let values = coeff_values(e, x, p.integer(0), 5, &p);
        assert_close(&values[lead..], &[-1.0 / 6.0, 0.0, -7.0 / 360.0, 0.0]);
        let general = without_tps(|| local_expansion(e, x, p.integer(0), 5, &p).unwrap());
        assert_eq!(general.valuation, 1);
    }

    /// `(x²−1)/(x−1)` about `x = 1`: the singular factor cancels outright, so
    /// there is no series left to divide — only a polynomial to expand.
    #[test]
    fn a_singularity_that_cancels_outright_expands() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let n = p.add(vec![p.pow(x, p.integer(2)), p.integer(-1)]);
        let d = p.add(vec![x, p.integer(-1)]);
        let e = p.mul(vec![n, p.pow(d, p.integer(-1))]);
        // (x²−1)/(x−1) = x + 1 = 2 + (x−1)
        assert_close(
            &coeff_values(e, x, p.integer(1), 4, &p),
            &[2.0, 1.0, 0.0, 0.0],
        );
    }

    /// A genuine pole is not a removable singularity and must keep its
    /// principal part: `1/sin x = x⁻¹ + x/6 + 7x³/360 + …`.
    #[test]
    fn a_real_pole_keeps_its_principal_part() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let s = p.func("sin", vec![x]);
        let e = p.pow(s, p.integer(-1));
        let exp = local_expansion(e, x, p.integer(0), 5, &p).unwrap();
        assert_eq!(exp.valuation, -1);
        let env = HashMap::new();
        let vals: Vec<f64> = exp
            .coeffs
            .iter()
            .map(|&c| crate::jit::eval_interp(c, &env, &p).unwrap_or(f64::NAN))
            .collect();
        assert_close(&vals, &[1.0, 0.0, 1.0 / 6.0, 0.0, 7.0 / 360.0]);
    }

    /// A free parameter survives the repair: `sin(kx)/x = k − k³x²/6 + O(x⁴)`.
    /// The coefficients are symbolic, so the indeterminacy check must not read
    /// "cannot evaluate" as "not a number".
    #[test]
    fn a_symbolic_parameter_survives_the_repair() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let k = p.symbol("k", Domain::Real);
        let s = p.func("sin", vec![p.mul(vec![k, x])]);
        let e = p.mul(vec![s, p.pow(x, p.integer(-1))]);
        let exp = local_expansion(e, x, p.integer(0), 3, &p).unwrap();
        assert_eq!(exp.valuation, 0);
        let mut env = HashMap::new();
        env.insert(k, 2.0);
        let vals: Vec<f64> = exp
            .coeffs
            .iter()
            .map(|&c| crate::jit::eval_interp(c, &env, &p).unwrap_or(f64::NAN))
            .collect();
        assert_close(&vals, &[2.0, 0.0, -8.0 / 6.0]);
        assert!(series(e, x, p.integer(0), 3, &p).is_ok());
        assert_eq!(take_series_refusal(), None);
    }

    // -----------------------------------------------------------------------
    // Singularities that are *not* removable: refuse, never fabricate
    // -----------------------------------------------------------------------

    fn assert_refuses_as_indeterminate(expr: ExprId, var: ExprId, order: u32, p: &ExprPool) {
        use crate::errors::AlkahestError;
        let err = series(expr, var, p.integer(0), order, p)
            .err()
            .unwrap_or_else(|| panic!("expected a refusal, got a Series"));
        assert!(matches!(err, SeriesError::InvalidOrder), "{err:?}");
        let refusal = take_series_refusal().expect("indeterminate-coefficient refusal recorded");
        assert_eq!(refusal.code(), "E-SERIES-004");
        assert_eq!(
            refusal.cause(),
            SeriesRefusalCause::IndeterminateCoefficient
        );
    }

    /// `√x`, `log x`, `e^{1/x}`, `x^x` and `x·sin(1/x)` have no Laurent
    /// expansion at `0` — branch point, logarithmic singularity, essential
    /// singularity, and two shapes that are not meromorphic at all. Every one
    /// of them used to come back as a *successful* `Series` whose coefficients
    /// were `sqrt(0)⁻¹`, `log(0)`, `exp(0⁻¹)`, …: `Ok`, unevaluable, `NaN` on
    /// contact. A coded refusal is the only honest answer, and it is what the
    /// caller can branch on.
    #[test]
    fn a_singularity_with_no_laurent_expansion_is_refused() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let inv_x = p.pow(x, p.integer(-1));
        let log_x = p.func("log", vec![x]);

        assert_refuses_as_indeterminate(p.func("sqrt", vec![x]), x, 4, &p);
        assert_refuses_as_indeterminate(log_x, x, 4, &p);
        assert_refuses_as_indeterminate(p.mul(vec![x, log_x]), x, 4, &p);
        assert_refuses_as_indeterminate(p.mul(vec![log_x, inv_x]), x, 4, &p);
        assert_refuses_as_indeterminate(p.func("exp", vec![inv_x]), x, 4, &p);
        assert_refuses_as_indeterminate(p.func("sin", vec![inv_x]), x, 4, &p);
        let x_sin_inv = p.mul(vec![x, p.func("sin", vec![inv_x])]);
        assert_refuses_as_indeterminate(x_sin_inv, x, 4, &p);
        let x_to_x = p.func("exp", vec![p.mul(vec![x, log_x])]);
        assert_refuses_as_indeterminate(x_to_x, x, 3, &p);
        // Puiseux, not Laurent: a half-integer valuation.
        let sqrt_sin = p.func("sqrt", vec![p.func("sin", vec![x])]);
        assert_refuses_as_indeterminate(sqrt_sin, x, 4, &p);
    }

    /// A free parameter must not hide the indeterminate part.
    ///
    /// The `x¹` coefficient of `k·√x` is `k · ½ · sqrt(0)⁻¹`. Evaluating the
    /// coefficient as a whole gets nowhere — `k` has no value — so a check that
    /// only looked at the top node passed it, and `k·√x` came back as a
    /// successful `Series` carrying `+∞`. The check has to look *inside*.
    #[test]
    fn a_free_parameter_does_not_hide_an_indeterminate_coefficient() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let k = p.symbol("k", Domain::Real);
        let sqrt_x = p.func("sqrt", vec![x]);

        assert_refuses_as_indeterminate(p.mul(vec![k, sqrt_x]), x, 3, &p);
        assert_refuses_as_indeterminate(p.mul(vec![k, p.func("log", vec![x])]), x, 3, &p);
        let k_sqrt_sin = p.mul(vec![k, sqrt_x, p.func("sin", vec![x])]);
        assert_refuses_as_indeterminate(k_sqrt_sin, x, 4, &p);

        // The control: a free parameter over a *pole* is an ordinary Laurent
        // expansion and must still succeed.
        let k_over_x = p.mul(vec![k, p.pow(x, p.integer(-1))]);
        assert!(series(k_over_x, x, p.integer(0), 3, &p).is_ok());
        assert_eq!(take_series_refusal(), None);
    }

    /// The class-level gate. Every expansion in the corpus either refuses or
    /// returns a `Series` in which **no** coefficient evaluates to `NaN` or
    /// `±∞` and none contains a `0^{negative}` node.
    ///
    /// This is the invariant, not the individual answers: a `Series` that
    /// reports success and cannot be evaluated is the failure mode, whatever
    /// produced it.
    #[test]
    fn no_successful_series_ever_carries_an_indeterminate_coefficient() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let zero = p.integer(0);
        let one = p.integer(1);
        let inv_x = p.pow(x, p.integer(-1));
        let sin_x = p.func("sin", vec![x]);
        let cos_x = p.func("cos", vec![x]);
        let exp_x = p.func("exp", vec![x]);
        let log_x = p.func("log", vec![x]);
        let x_minus_1 = p.add(vec![x, p.integer(-1)]);
        let one_minus_cos = p.add(vec![one, p.mul(vec![p.integer(-1), cos_x])]);

        let cases: Vec<(ExprId, ExprId, u32)> = vec![
            // removable
            (p.mul(vec![sin_x, inv_x]), zero, 6),
            (p.mul(vec![p.func("tan", vec![x]), inv_x]), zero, 5),
            (p.mul(vec![p.func("sinh", vec![x]), inv_x]), zero, 5),
            (p.mul(vec![p.func("atan", vec![x]), inv_x]), zero, 5),
            (p.mul(vec![one_minus_cos, p.pow(x, p.integer(-2))]), zero, 5),
            (
                p.mul(vec![p.add(vec![exp_x, p.integer(-1)]), inv_x]),
                zero,
                5,
            ),
            (p.mul(vec![x, p.pow(sin_x, p.integer(-1))]), zero, 5),
            (
                p.mul(vec![p.pow(sin_x, p.integer(2)), p.pow(x, p.integer(-2))]),
                zero,
                5,
            ),
            (
                p.add(vec![
                    inv_x,
                    p.mul(vec![p.integer(-1), p.pow(sin_x, p.integer(-1))]),
                ]),
                zero,
                5,
            ),
            // poles — legitimate Laurent expansions
            (inv_x, zero, 4),
            (p.pow(x, p.integer(-2)), zero, 4),
            (p.pow(sin_x, p.integer(-1)), zero, 5),
            (p.mul(vec![cos_x, p.pow(sin_x, p.integer(-1))]), zero, 5),
            (p.mul(vec![sin_x, p.pow(x, p.integer(-3))]), zero, 5),
            (p.pow(one_minus_cos, p.integer(-1)), zero, 4),
            (p.pow(p.func("tan", vec![x]), p.integer(-1)), zero, 4),
            // non-zero expansion points, regular and singular
            (exp_x, one, 5),
            (log_x, one, 5),
            (p.func("sqrt", vec![x]), one, 4),
            (p.pow(log_x, p.integer(-1)), one, 4),
            (
                p.mul(vec![
                    p.func("sqrt", vec![x]),
                    p.pow(x_minus_1, p.integer(-1)),
                ]),
                one,
                4,
            ),
            (
                p.pow(
                    p.add(vec![p.pow(x, p.integer(2)), p.integer(-1)]),
                    p.integer(-1),
                ),
                one,
                4,
            ),
            (
                p.mul(vec![
                    p.add(vec![p.pow(x, p.integer(2)), p.integer(-1)]),
                    p.pow(x_minus_1, p.integer(-1)),
                ]),
                one,
                4,
            ),
            // branch points / essential singularities
            (p.func("sqrt", vec![x]), zero, 4),
            (log_x, zero, 4),
            (p.func("exp", vec![inv_x]), zero, 4),
            (p.func("sin", vec![inv_x]), zero, 4),
            (p.func("gamma", vec![x]), zero, 3),
            // controls
            (exp_x, zero, 6),
            (
                p.pow(
                    p.add(vec![one, p.mul(vec![p.integer(-1), x])]),
                    p.integer(-1),
                ),
                zero,
                6,
            ),
            (p.func("sqrt", vec![p.add(vec![one, x])]), zero, 5),
            (p.func("tan", vec![x]), zero, 8),
        ];

        let env = HashMap::new();
        for (i, &(e, point, order)) in cases.iter().enumerate() {
            let Ok(s) = series(e, x, point, order, &p) else {
                let _ = take_series_refusal();
                continue;
            };
            assert_eq!(take_series_refusal(), None, "case {i}: refusal left behind");
            let mut stack = match p.get(s.expr()) {
                ExprData::Add(xs) => xs,
                _ => vec![s.expr()],
            };
            while let Some(t) = stack.pop() {
                if matches!(p.get(t), ExprData::BigO(_)) {
                    continue;
                }
                assert!(
                    !contains_zero_to_negative_power(t, &p),
                    "case {i}: {} contains 0^-n",
                    p.display(s.expr())
                );
                if let Some(v) = crate::jit::eval_interp(t, &env, &p) {
                    assert!(
                        v.is_finite(),
                        "case {i}: term {} of {} evaluates to {v}",
                        p.display(t),
                        p.display(s.expr())
                    );
                }
                // Descend so a non-finite factor inside a term that still has
                // a free `x` in it cannot hide.
                match p.get(t) {
                    ExprData::Add(xs) | ExprData::Mul(xs) => stack.extend(xs),
                    ExprData::Pow { base, exp } => {
                        stack.push(base);
                        stack.push(exp);
                    }
                    ExprData::Func { args, .. } => stack.extend(args),
                    _ => {}
                }
            }
        }
    }

    /// `limit` reads coefficients out of the same expansion, so the repair has
    /// to leave the standard limits right. `lim_{x→0} sin(x)/x = 1` now comes
    /// straight off the constant term instead of falling through to L'Hôpital.
    #[test]
    fn the_repair_agrees_with_the_limit_engine() {
        use crate::calculus::limits::{limit, LimitDirection};
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let e = sin_over_x(&p);
        let l = limit(e, x, p.integer(0), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(crate::jit::eval_interp(l, &HashMap::new(), &p), Some(1.0));
    }
}

/// The truncated-power-series fast path against the general route it
/// replaces, in one pool: same `ExprId` (hence the same terms, the same `O(·)`
/// and the same coefficient literals), or the same error.
#[cfg(test)]
mod tps_differential {
    use super::*;
    use crate::kernel::Domain;

    /// `Ok(id)` / `Err(code)` — the refusal code when there is one.
    fn run(
        e: ExprId,
        x: ExprId,
        point: ExprId,
        order: u32,
        p: &ExprPool,
    ) -> Result<ExprId, String> {
        use crate::errors::AlkahestError;
        // The general route has pathological inputs of its own (a random
        // composition can make one coefficient's `simplify` run for minutes),
        // so every run gets a wall clock; a trip reads `E-BUDGET`.
        let _budget = crate::budget::enter(
            crate::budget::Budget::new().with_wall(std::time::Duration::from_secs(5)),
        );
        match series(e, x, point, order, p) {
            Ok(s) => Ok(s.expr()),
            Err(err) => Err(match take_series_refusal() {
                Some(r) if r.budget().is_some() => "E-BUDGET".to_string(),
                Some(r) => r.code().to_string(),
                None => err.code().to_string(),
            }),
        }
    }

    /// How the fast path's result relates to the general route's.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Outcome {
        /// The fast path declined; the general route ran (trivially the same).
        Declined,
        /// The same `ExprId`.
        Same,
        /// Same terms and `O(·)`, but the general route left a coefficient
        /// unevaluated — `atanh(0)`, `sin(atanh(0))` — that `simplify` does
        /// not fold and that evaluates to the fast path's rational.
        Canonicalized,
        /// The general route refused (`E-SERIES-003`/`004`) a series the fast
        /// path computes.
        Recovered,
        /// A difference the checks above do not explain. Never allowed in
        /// the committed corpora; in a survey, dumped for an external oracle
        /// (every one found so far was a general-route error — see
        /// `the_general_route_was_wrong_here`).
        Suspect,
    }

    /// `{exponent: value}` and the `O(·)` exponent of a series in `h`, with
    /// each coefficient evaluated numerically; also whether every
    /// coefficient was already a rational literal. `None` when a coefficient
    /// cannot be evaluated.
    fn numeric_terms(
        s: ExprId,
        h: ExprId,
        p: &ExprPool,
    ) -> Option<(std::collections::BTreeMap<i64, f64>, i64, bool)> {
        let power = |f: ExprId| -> Option<i64> {
            if f == h {
                return Some(1);
            }
            match p.get(f) {
                ExprData::Pow { base, exp } if base == h => match p.get(exp) {
                    ExprData::Integer(n) => n.0.to_i64(),
                    _ => None,
                },
                _ => None,
            }
        };
        let parts = match p.get(s) {
            ExprData::Add(xs) => xs,
            _ => vec![s],
        };
        let env = HashMap::new();
        let mut terms = std::collections::BTreeMap::new();
        let (mut big_o, mut literal) = (None, true);
        for t in parts {
            if let ExprData::BigO(arg) = p.get(t) {
                big_o = power(arg);
                continue;
            }
            let factors = match p.get(t) {
                ExprData::Mul(xs) => xs,
                _ => vec![t],
            };
            let (mut k, mut v) = (0_i64, 1.0_f64);
            for f in factors {
                match power(f) {
                    Some(e) => k += e,
                    None => {
                        literal &= matches!(p.get(f), ExprData::Integer(_) | ExprData::Rational(_));
                        v *= crate::jit::eval_interp(f, &env, p)?;
                    }
                }
            }
            *terms.entry(k).or_insert(0.0) += v;
        }
        terms.retain(|_, v| *v != 0.0);
        Some((terms, big_o?, literal))
    }

    /// `e` in Wolfram Language syntax, so a surveyed case can be checked
    /// against `Series` directly.
    fn to_wl(e: ExprId, p: &ExprPool) -> String {
        let join = |xs: &[ExprId], sep: &str| {
            xs.iter()
                .map(|&a| format!("({})", to_wl(a, p)))
                .collect::<Vec<_>>()
                .join(sep)
        };
        match p.get(e) {
            ExprData::Integer(n) => format!("({})", n.0),
            ExprData::Rational(r) => format!("({}/{})", r.0.numer(), r.0.denom()),
            ExprData::Symbol { name, .. } => name,
            ExprData::Add(xs) => join(&xs, "+"),
            ExprData::Mul(xs) => join(&xs, "*"),
            ExprData::Pow { base, exp } => format!("({})^({})", to_wl(base, p), to_wl(exp, p)),
            ExprData::Func { name, args } => {
                let head = match name.as_str() {
                    "sin" => "Sin",
                    "cos" => "Cos",
                    "tan" => "Tan",
                    "exp" => "Exp",
                    "log" => "Log",
                    "sqrt" => "Sqrt",
                    "atan" => "ArcTan",
                    "asin" => "ArcSin",
                    "acos" => "ArcCos",
                    "sinh" => "Sinh",
                    "cosh" => "Cosh",
                    "tanh" => "Tanh",
                    "atanh" => "ArcTanh",
                    "asinh" => "ArcSinh",
                    other => other,
                };
                format!("{head}[{}]", join(&args, ","))
            }
            ExprData::BigO(_) => "0".to_string(),
            other => format!("Unsupported[\"{other:?}\"]"),
        }
    }

    /// One surveyed case, for `ALKAHEST_TPS_DIFF_SURVEY`: tab-separated
    /// `kind, f, point, order, fast-path series without its O(·)`.
    fn survey_line(kind: &str, e: ExprId, point: ExprId, order: u32, s: ExprId, p: &ExprPool) {
        let parts = match p.get(s) {
            ExprData::Add(xs) => xs,
            _ => vec![s],
        };
        let kept: Vec<_> = parts
            .into_iter()
            .filter(|&t| !matches!(p.get(t), ExprData::BigO(_)))
            .collect();
        let poly = if kept.is_empty() {
            "0".to_string()
        } else {
            kept.iter()
                .map(|&t| format!("({})", to_wl(t, p)))
                .collect::<Vec<_>>()
                .join("+")
        };
        eprintln!(
            "SURVEY\t{kind}\t{}\t{}\t{order}\t{poly}",
            to_wl(e, p),
            to_wl(point, p)
        );
    }

    /// Fast path vs general route for one case.
    ///
    /// Panics unless the results are the same `ExprId` or one of the
    /// documented differences: [`Outcome::Canonicalized`] (checked
    /// numerically term by term), [`Outcome::Recovered`] (checked against
    /// every lower order the general route reaches), and
    /// [`Outcome::Suspect`] — a disagreement neither check explains, which
    /// fails the corpus unless it is being *surveyed* for an external oracle.
    fn check(e: ExprId, x: ExprId, point: ExprId, order: u32, p: &ExprPool) -> Outcome {
        let old = without_tps(|| run(e, x, point, order, p));
        let new = run(e, x, point, order, p);
        let xi = p.symbol("__sxp", Domain::Real);
        let mut m = HashMap::new();
        m.insert(x, p.add(vec![point, xi]));
        let fast = super::super::tps::tps_expansion(subs(e, &m, p), xi, order, p).is_some();
        let shown = |r: &Result<ExprId, String>| match r {
            Ok(id) => p.display(*id).to_string(),
            Err(c) => c.clone(),
        };
        let what = || {
            format!(
                "series({}, order {order}) at {}: general route {} vs fast path {}",
                p.display(e),
                p.display(point),
                shown(&old),
                shown(&new),
            )
        };
        if old == new {
            return if fast {
                Outcome::Same
            } else {
                Outcome::Declined
            };
        }
        let budget_trip = |r: &Result<ExprId, String>| matches!(r, Err(c) if c == "E-BUDGET");
        if !fast && (budget_trip(&old) || budget_trip(&new)) {
            // The general route twice, against a wall clock: timing, not a
            // difference.
            return Outcome::Declined;
        }
        assert!(
            fast,
            "the fast path declined and still changed the result: {}",
            what()
        );
        let n = *new.as_ref().unwrap_or_else(|_| panic!("{}", what()));
        let h = expansion_increment(p, x, point);
        let (nt, no, _) = numeric_terms(n, h, p).expect("fast path is rational");
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
        let suspect = || {
            eprintln!("suspect: {}", what());
            survey_line("suspect", e, point, order, n, p);
            Outcome::Suspect
        };
        match &old {
            Ok(o) => {
                let Some((ot, oo, o_literal)) = numeric_terms(*o, h, p) else {
                    return suspect();
                };
                let same_terms =
                    oo == no && ot.keys().eq(nt.keys()) && ot.iter().all(|(k, v)| close(*v, nt[k]));
                if o_literal || !same_terms {
                    return suspect();
                }
                eprintln!("canonicalized: {}", what());
                Outcome::Canonicalized
            }
            Err(code) => {
                assert!(
                    matches!(code.as_str(), "E-SERIES-003" | "E-SERIES-004" | "E-BUDGET"),
                    "the general route failed with {code}: {}",
                    what()
                );
                // Every lower order the general route reaches must be a
                // prefix of the recovered series.
                for lower in 1..order {
                    if let Ok(o) = without_tps(|| run(e, x, point, lower, p)) {
                        let Some((ot, _, _)) = numeric_terms(o, h, p) else {
                            return suspect();
                        };
                        let want: Vec<_> =
                            nt.iter().filter(|(k, _)| **k < i64::from(lower)).collect();
                        let agree = ot.len() == want.len()
                            && ot
                                .iter()
                                .zip(want)
                                .all(|((gk, gv), (wk, wv))| gk == wk && close(*gv, *wv));
                        if !agree {
                            return suspect();
                        }
                    }
                }
                eprintln!("recovered: {}", what());
                survey_line("recovered", e, point, order, n, p);
                Outcome::Recovered
            }
        }
    }

    /// Run a corpus and report how each case came out.
    fn tally(outcomes: &[Outcome]) -> [usize; 5] {
        let mut t = [0; 5];
        for o in outcomes {
            t[*o as usize] += 1;
        }
        t
    }

    fn assert_series(e: ExprId, x: ExprId, order: u32, want: &[(i64, i64, i64)], p: &ExprPool) {
        let s = series(e, x, p.integer(0), order, p)
            .unwrap_or_else(|err| panic!("{}: {err}", p.display(e)))
            .expr();
        let (terms, big_o, literal) = numeric_terms(s, x, p).expect("evaluable");
        assert!(literal, "{}", p.display(s));
        assert_eq!(big_o, i64::from(order), "{}", p.display(s));
        let want: Vec<(i64, f64)> = want
            .iter()
            .map(|&(k, n, d)| (k, n as f64 / d as f64))
            .collect();
        let got: Vec<(i64, f64)> = terms.into_iter().collect();
        assert_eq!(got.len(), want.len(), "{}", p.display(s));
        for ((gk, gv), (wk, wv)) in got.iter().zip(&want) {
            assert_eq!(gk, wk, "{}", p.display(s));
            assert!((gv - wv).abs() <= 1e-15 * wv.abs(), "{}", p.display(s));
        }
    }

    /// Cases the random differential found where the general route was
    /// wrong or refused, checked against Mathematica 14 `Series`.
    #[test]
    fn the_general_route_was_wrong_here() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let tanh = p.func("tanh", vec![x]);
        // x·tanh(x)⁻³ = x⁻² + 1 + 4x²/15 − x⁴/945 − 11x⁶/4725 + …
        // The general route answered order 1 with a bare `O(x)` — both terms
        // below the order missing — and refused every higher order.
        let e = p.mul(vec![x, p.pow(tanh, p.integer(-3))]);
        assert_series(e, x, 1, &[(-2, 1, 1), (0, 1, 1)], &p);
        assert_series(
            e,
            x,
            7,
            &[
                (-2, 1, 1),
                (0, 1, 1),
                (2, 4, 15),
                (4, -1, 945),
                (6, -11, 4725),
            ],
            &p,
        );
        // log(1 + atanh x)⁻³: refused (E-SERIES-004) at every order.
        let l = p.func(
            "log",
            vec![p.add(vec![p.integer(1), p.func("atanh", vec![x])])],
        );
        assert_series(
            p.pow(l, p.integer(-3)),
            x,
            3,
            &[
                (-3, 1, 1),
                (-2, 3, 2),
                (-1, -1, 2),
                (0, -1, 1),
                (1, -23, 240),
                (2, -49, 480),
            ],
            &p,
        );
        // tanh x = x − x³/3 + 2x⁵/15: the general route's coefficients were
        // `tanh(0)`-laden (`1 + (−1·tanh(0)²)` for the x¹ term), which
        // `simplify` does not fold.
        assert_series(tanh, x, 6, &[(1, 1, 1), (3, -1, 3), (5, 2, 15)], &p);
    }

    /// The f⁻ᵏ grid of `tests/test_series_negative_valuation.py` (#435), whose
    /// coefficients are checked there against Mathematica.
    #[test]
    fn negative_power_grid_is_unchanged() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let one = p.integer(1);
        let neg = |e: ExprId| p.mul(vec![p.integer(-1), e]);
        let fs = [
            p.func("sin", vec![x]),
            p.func("tan", vec![x]),
            p.func("asin", vec![x]),
            p.func("atan", vec![x]),
            p.func("sinh", vec![x]),
            p.add(vec![x, p.mul(vec![x, x])]),
            p.add(vec![p.func("exp", vec![x]), p.integer(-1)]),
            p.func("log", vec![p.add(vec![one, x])]),
            p.add(vec![one, neg(p.func("cos", vec![x]))]),
            p.mul(vec![p.add(vec![x, p.integer(-2)]), p.pow(x, p.integer(-1))]),
            p.mul(vec![x, p.func("cos", vec![x])]),
        ];
        let mut outcomes = Vec::new();
        for f in fs {
            for k in 1..=6 {
                for order in 1..=5 {
                    let e = p.pow(f, p.integer(-k));
                    outcomes.push(check(e, x, p.integer(0), order, &p));
                }
            }
        }
        // Every case is in the fast path's domain, and identical.
        assert_eq!(tally(&outcomes), [0, 330, 0, 0, 0]);
    }

    /// The existing suite's expansions, and the benchmark shapes.
    #[test]
    fn named_cases_are_unchanged() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let a = p.symbol("a", Domain::Real);
        let (zero, one, two) = (p.integer(0), p.integer(1), p.integer(2));
        let neg = |e: ExprId| p.mul(vec![p.integer(-1), e]);
        let inv = |e: ExprId| p.pow(e, p.integer(-1));
        let f = |n: &str, e: ExprId| p.func(n, vec![e]);
        let sin_tan = p.add(vec![f("sin", f("tan", x)), neg(f("tan", f("sin", x)))]);
        let cases: Vec<(ExprId, ExprId, u32)> = vec![
            (sin_tan, zero, 15),
            (p.mul(vec![sin_tan, p.pow(x, p.integer(-7))]), zero, 4),
            (f("exp", f("sin", x)), zero, 20),
            (inv(f("cos", x)), zero, 20),
            (
                p.mul(vec![
                    f("log", p.add(vec![one, x])),
                    inv(p.add(vec![one, neg(x)])),
                ]),
                zero,
                30,
            ),
            (f("sin", x), zero, 24),
            (f("tan", x), zero, 16),
            (f("sqrt", p.add(vec![one, x])), zero, 24),
            (p.mul(vec![f("sin", x), inv(x)]), zero, 6),
            (p.mul(vec![f("tan", x), inv(x)]), zero, 6),
            (inv(f("sin", x)), zero, 3),
            (p.add(vec![inv(x), neg(inv(f("sin", x)))]), zero, 5),
            (p.pow(f("sin", x), p.integer(-4)), zero, 4),
            (
                p.pow(p.add(vec![x, p.mul(vec![x, x])]), p.integer(-3)),
                zero,
                3,
            ),
            (
                inv(p.mul(vec![p.add(vec![x, p.integer(-2)]), inv(x)])),
                zero,
                4,
            ),
            (p.mul(vec![x, p.add(vec![inv(x), one])]), zero, 3),
            (
                f(
                    "exp",
                    p.mul(vec![x, inv(p.mul(vec![p.integer(-2), inv(x)]))]),
                ),
                zero,
                5,
            ),
            (
                p.mul(vec![
                    f("log", p.mul(vec![x, inv(x)])),
                    p.add(vec![x, p.pow(x, p.integer(3))]),
                ]),
                zero,
                3,
            ),
            (
                p.mul(vec![
                    two,
                    x,
                    f("exp", x),
                    inv(p.mul(vec![p.add(vec![x, p.integer(-2)]), inv(f("asin", x))])),
                ]),
                zero,
                4,
            ),
            // Expansion away from 0 and outside ℚ: the general route decides.
            (f("log", x), one, 6),
            (inv(x), two, 5),
            (f("sin", x), one, 5),
            (f("exp", p.mul(vec![a, x])), zero, 5),
            (f("sqrt", x), zero, 4),
            (f("log", x), zero, 4),
            (f("exp", inv(x)), zero, 4),
            (p.mul(vec![x, x, inv(f("log", x))]), zero, 4),
            (f("acos", x), zero, 5),
            (f("atanh", x), zero, 9),
            (f("asinh", x), zero, 9),
            (f("cosh", f("tanh", x)), zero, 9),
            (f("sqrt", p.add(vec![p.integer(4), x])), zero, 6),
            (f("sqrt", p.add(vec![two, x])), zero, 4),
            (p.pow(p.add(vec![one, x]), p.rational(2, 3)), zero, 6),
            (p.pow(p.add(vec![one, x]), p.rational(-5, 2)), zero, 6),
        ];
        let outcomes: Vec<_> = cases
            .iter()
            .map(|&(e, point, order)| check(e, x, point, order, &p))
            .collect();
        let [declined, same, canonicalized, recovered, suspect] = tally(&outcomes);
        eprintln!(
            "declined {declined} same {same} canonicalized {canonicalized}              recovered {recovered} suspect {suspect}"
        );
        assert_eq!(suspect, 0, "{outcomes:?}");
        assert!(same >= 20, "{outcomes:?}");
    }

    fn build(p: &ExprPool, x: ExprId, depth: u32, next: &mut dyn FnMut(u64) -> u64) -> ExprId {
        if depth == 0 || next(4) == 0 {
            return match next(4) {
                0 | 1 => x,
                2 => p.integer(next(5) as i64 - 2),
                _ => p.rational(next(7) as i64 - 3, 2 + next(3) as i64),
            };
        }
        let a = build(p, x, depth - 1, next);
        match next(14) {
            0 | 1 => {
                let b = build(p, x, depth - 1, next);
                p.add(vec![a, b])
            }
            2 | 3 => {
                let b = build(p, x, depth - 1, next);
                p.mul(vec![a, b])
            }
            4 => p.pow(a, p.integer(next(6) as i64 - 3)),
            5 => p.pow(
                p.add(vec![p.integer(1), a]),
                p.rational(next(5) as i64 - 2, 2),
            ),
            _ => {
                const HEADS: [&str; 13] = [
                    "sin", "cos", "tan", "exp", "log", "atan", "asin", "sinh", "cosh", "tanh",
                    "atanh", "asinh", "sqrt",
                ];
                let h = HEADS[next(HEADS.len() as u64) as usize];
                let arg = if matches!(h, "log" | "sqrt") && next(3) != 0 {
                    p.add(vec![p.integer(1), a])
                } else {
                    a
                };
                p.func(h, vec![arg])
            }
        }
    }

    /// Random compositions of the heads the fast path covers (and of shapes
    /// it declines), at `0` and `1`. `ALKAHEST_TPS_DIFF_N` widens the corpus
    /// and `ALKAHEST_TPS_DIFF_SEED` moves it.
    #[test]
    fn random_compositions_are_unchanged() {
        let env = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<u64>().ok());
        let n = env("ALKAHEST_TPS_DIFF_N").unwrap_or(150) as usize;
        let mut state: u64 = 0x2545_f491_4f6c_dd1d ^ env("ALKAHEST_TPS_DIFF_SEED").unwrap_or(0);
        let mut next = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let mut outcomes = Vec::new();
        for _ in 0..n {
            let e = build(&p, x, 3, &mut next);
            let point = if next(5) == 0 {
                p.integer(1)
            } else {
                p.integer(0)
            };
            let order = 1 + next(7) as u32;
            outcomes.push(check(e, x, point, order, &p));
        }
        let [declined, same, canonicalized, recovered, suspect] = tally(&outcomes);
        eprintln!(
            "declined {declined} same {same} canonicalized {canonicalized}              recovered {recovered} suspect {suspect}"
        );
        if std::env::var_os("ALKAHEST_TPS_DIFF_SURVEY").is_none() {
            assert_eq!(suspect, 0, "{outcomes:?}");
        }
        assert!((same + canonicalized) * 5 >= n, "{outcomes:?}");
    }
}
