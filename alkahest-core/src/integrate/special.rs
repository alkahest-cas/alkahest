//! Non-elementary antiderivatives over a **registered special-function basis**.
//!
//! # Why this module exists
//!
//! Before it, [`crate::integrate::integrate`] had exactly two shapes of answer:
//! an elementary closed form, or a refusal.  The `EllipticF`/`EllipticE` output
//! in [`crate::integrate::algebraic::elliptic_output`] was the sole exception,
//! and it was reached only from the algebraic engine.  Everything else —
//! `∫eˣ/x`, `∫sin(x)/x`, `∫exp(−x²)`, `∫sin(x²)` — came back as `E-INT-004`
//! *"no elementary antiderivative exists"*, which is **true** and **useless**:
//! `erf`, `Ei`, `Si`, `Ci`, `Shi`, `Chi`, `li`, `S`, `C` and `Li₂` have all been
//! complete primitives (derivative + `f64` kernel + ball kernel + Taylor rule)
//! for as long as the refusal has been there.  The integrator knew the names and
//! would not say them.
//!
//! `engine::known_nonelementary` is the sharpest illustration: it *already*
//! recognises the Ei/Si/Ci/Shi/Chi/li shapes, purely in order to refuse them.
//! This module reuses the same recognition to **emit**.
//!
//! # The three-valued answer
//!
//! `planning/risch.md` §4.3 asks for
//!
//! ```text
//! Elementary(F)                      — as before
//! NonElementaryClosedForm(F, basis)  — F names special functions, still gate-verified
//! NonElementary(reason)              — no closed form in the extended basis either
//! ```
//!
//! [`IntegrationAnswer`] and [`classify`] provide the first two.  **The third is
//! deliberately left alone**, and the reason is the whole point of the exercise:
//!
//! > Strengthening `NonElementary` from *"not elementary"* to *"not elementary
//! > **and** not expressible over the registered basis"* would be a **new and
//! > unearned claim**.  Nothing in this codebase decides expressibility over the
//! > basis; this module is a table of recognised shapes, and a table's silence
//! > is not a theorem.  `∫sin(x)/x² dx` is the standing counterexample — it is
//! > non-elementary (so today's certificate is sound), it *is* expressible as
//! > `−sin(x)/x + Ci(x)`, and no matcher here finds that.  Re-reading the
//! > existing certificate as the stronger statement would manufacture a false
//! > one, which is exactly the defect eight families of this codebase were
//! > already found to have.
//!
//! So [`IntegrationError::NonElementary`] keeps precisely the meaning and the
//! wording it has always had: *no **elementary** antiderivative exists*.  The
//! strengthening is available to whoever builds a decision procedure for the
//! extended basis, and this module's docs are the place to record that it is not
//! built.
//!
//! # Soundness
//!
//! Every candidate goes through [`verify_antiderivative_status`] — symbolic
//! `d/dx F − f ≡ 0` first, then the in-domain `f64` screen — before it is
//! returned, exactly as `elliptic_output`, `by_parts` and `norman` do.  A
//! candidate that cannot be confirmed is discarded and the caller falls through
//! to whatever verdict it already had.  There is **no path from this module to
//! `NonElementary`**: it returns `Some(F)` or `None`, and `None` is a decline.
//!
//! # Where it runs
//!
//! Two kinds of site in [`crate::integrate::engine::integrate`], both after the
//! elementary pipeline has had its turn:
//!
//! 1. wherever a `NonElementary` verdict would otherwise leave `integrate` —
//!    the `known_nonelementary` pre-check and every sub-engine exit, funnelled
//!    through `engine::emit_or_keep`.  This is the load-bearing one: the whole
//!    exponential family (`∫eˣ/x`, `∫exp(−x²)`) is decided by the Risch tower,
//!    which returns `NonElementary` and short-circuits, so an emitter placed
//!    only on the `NotImplemented` path never sees them.
//! 2. on the `NotImplemented` cascade, for the shapes no engine has a verdict
//!    on (`∫sin(x²)`, `∫log x/(1+x)`).
//!
//! Never before the rule engine: a non-elementary answer must not pre-empt an
//! elementary one.  And never *overturning* the verdict at site 1 — the verdict
//! is that no **elementary** antiderivative exists, which stays true; what is
//! answered is the strictly weaker question the verdict did not settle.
//!
//! # Scope
//!
//! | integrand | answer |
//! |---|---|
//! | `c·f(g)/d`, `f ∈ {exp,sin,cos,sinh,cosh}`, `g`,`d` linear, `d ∝ g` | `Ei`/`Si`/`Ci`/`Shi`/`Chi` |
//! | `c·exp(a·x+b)/(p·x)` | `Ei` (the `e^b` is pulled out) |
//! | `c/log(g)`, `g` linear | `li` |
//! | `c·exp(A·x²+B·x+C)`, `A < 0` | `erf` |
//! | `c·sin(A·x²)` / `c·cos(A·x²)` | Fresnel `S` / `C` |
//! | `c·log(x)/(a+b·x)`, `a·b ≠ 0` | `Li₂` |
//!
//! Not covered, and honest about it: `A > 0` in the Gaussian (would need
//! `erfi`, which is not a registered primitive), higher denominator powers
//! (`sin(x)/x²`), and non-linear arguments other than the pure quadratics above.

use std::collections::HashMap;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::engine::{
    is_free_of, is_linear_in, special_integral_name, verify_antiderivative_status,
    AntiderivativeVerification, IntegrationError,
};

// ---------------------------------------------------------------------------
// The basis
// ---------------------------------------------------------------------------

/// The named special functions an antiderivative may be expressed over.
///
/// Membership means three things, all of which are checked by the primitive
/// registry: the name has a derivative rule (so the gate can differentiate an
/// answer carrying it), an `f64` kernel (so the numeric screen can run), and a
/// registry entry (so `diff` will not refuse).
///
/// `EllipticPi` is **absent on purpose**.  It differentiates and evaluates, but
/// the validated tier's `Func` rules are unary and `Π` takes three arguments, so
/// an answer carrying it passes the symbolic half of the gate and can never pass
/// the rigorous half.  It is emitted by `elliptic_output` on its own authority;
/// it is not something this module will introduce.
pub const SPECIAL_BASIS: &[&str] = &[
    "Ei",
    "li",
    "Si",
    "Ci",
    "Shi",
    "Chi",
    "erf",
    "erfc",
    "fresnels",
    "fresnelc",
    "dilog",
    "EllipticF",
    "EllipticE",
    "EllipticK",
];

/// The basis functions actually named in `expr`, sorted and deduplicated.
///
/// Empty means the expression is elementary *as far as this basis is
/// concerned* — it does not certify elementarity, it reports vocabulary.
pub fn basis_functions_used(expr: ExprId, pool: &ExprPool) -> Vec<&'static str> {
    let mut found: Vec<&'static str> = Vec::new();
    collect_basis(expr, pool, &mut found, &mut Vec::new());
    found.sort_unstable();
    found.dedup();
    found
}

fn collect_basis(
    expr: ExprId,
    pool: &ExprPool,
    out: &mut Vec<&'static str>,
    seen: &mut Vec<ExprId>,
) {
    if seen.contains(&expr) {
        return;
    }
    seen.push(expr);
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            if let Some(hit) = SPECIAL_BASIS.iter().find(|&&b| b == name) {
                out.push(hit);
            }
            for a in args {
                collect_basis(a, pool, out, seen);
            }
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for a in args {
                collect_basis(a, pool, out, seen);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_basis(base, pool, out, seen);
            collect_basis(exp, pool, out, seen);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// The three-valued answer
// ---------------------------------------------------------------------------

/// A successful integration, split by the vocabulary the answer needs.
///
/// This is the `Ok` half of `planning/risch.md` §4.3's three-valued type.  The
/// `Err` half is [`IntegrationError`] and is unchanged — see the module docs for
/// why `NonElementary` was not re-read as the stronger claim.
#[derive(Debug, Clone)]
pub enum IntegrationAnswer {
    /// `F` is elementary: it names nothing from [`SPECIAL_BASIS`].
    Elementary(DerivedExpr<ExprId>),
    /// `F` names special functions and is still gate-verified by
    /// differentiation, exactly as the elliptic route already is.
    NonElementaryClosedForm {
        /// The antiderivative.
        antiderivative: DerivedExpr<ExprId>,
        /// Which of [`SPECIAL_BASIS`] it needs, sorted.
        basis: Vec<&'static str>,
    },
}

impl IntegrationAnswer {
    /// The antiderivative, whichever variant this is.
    pub fn antiderivative(&self) -> ExprId {
        match self {
            IntegrationAnswer::Elementary(d) => d.value,
            IntegrationAnswer::NonElementaryClosedForm { antiderivative, .. } => {
                antiderivative.value
            }
        }
    }

    /// The basis functions used; empty for [`IntegrationAnswer::Elementary`].
    pub fn basis(&self) -> &[&'static str] {
        match self {
            IntegrationAnswer::Elementary(_) => &[],
            IntegrationAnswer::NonElementaryClosedForm { basis, .. } => basis,
        }
    }

    /// `true` when the answer needs a name outside the elementary functions.
    pub fn is_non_elementary_closed_form(&self) -> bool {
        matches!(self, IntegrationAnswer::NonElementaryClosedForm { .. })
    }

    /// Discard the classification and keep the derived expression.
    pub fn into_derived(self) -> DerivedExpr<ExprId> {
        match self {
            IntegrationAnswer::Elementary(d) => d,
            IntegrationAnswer::NonElementaryClosedForm { antiderivative, .. } => antiderivative,
        }
    }
}

/// Classify an antiderivative returned by [`crate::integrate::integrate`].
///
/// Purely a vocabulary check on the returned expression; it runs no
/// mathematics and cannot change a verdict.
pub fn classify(result: DerivedExpr<ExprId>, pool: &ExprPool) -> IntegrationAnswer {
    let basis = basis_functions_used(result.value, pool);
    if basis.is_empty() {
        IntegrationAnswer::Elementary(result)
    } else {
        IntegrationAnswer::NonElementaryClosedForm {
            antiderivative: result,
            basis,
        }
    }
}

// ---------------------------------------------------------------------------
// The engine-facing hook
// ---------------------------------------------------------------------------

/// `Some((F, evidence))` with `d/dx F = f` already established, or `None`.
///
/// `None` is a decline and says nothing about the integrand.  Callers must not
/// convert it into a verdict.
pub fn try_special_antiderivative(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, AntiderivativeVerification)> {
    crate::budget::check().ok()?;

    // Work on the normalised copy: the parser does not build flat `Mul`/`Add`
    // nodes, so `sin(x)/x` arrives as a two-child `Mul` only after `simplify`.
    // The gate below still checks against the caller's own `expr`.
    let work = simplify(expr, pool).value;

    let candidates = [
        match_quotient_family(work, var, pool),
        match_log_reciprocal(work, var, pool),
        match_gaussian(work, var, pool),
        match_fresnel(work, var, pool),
        match_dilog(work, var, pool),
    ];

    for candidate in candidates.into_iter().flatten() {
        let f = simplify(candidate, pool).value;
        if let Some(evidence) = verify_antiderivative_status(f, expr, var, pool) {
            return Some((f, evidence));
        }
    }
    None
}

/// [`try_special_antiderivative`], packaged as the engine's return shape.
pub fn try_special_derived(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<DerivedExpr<ExprId>> {
    let (f, _evidence) = try_special_antiderivative(expr, var, pool)?;
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("special_function_integral", expr, f));
    Some(DerivedExpr::with_log(f, log))
}

/// Convert a decline into the engine's error type.  `NotImplemented` and
/// nothing else — see the module docs.
pub fn decline(expr: ExprId, pool: &ExprPool) -> IntegrationError {
    IntegrationError::NotImplemented(format!(
        "∫ {} — no closed form over the registered special-function basis",
        pool.display(expr)
    ))
}

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

/// The `f64` value of a `var`-free subexpression, or `None`.
fn const_f64(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<f64> {
    if !is_free_of(expr, var, pool) {
        return None;
    }
    if let Some(v) = crate::kernel::try_expr_f64(expr, pool) {
        return Some(v);
    }
    let env: HashMap<ExprId, f64> = HashMap::new();
    crate::jit::eval_interp(expr, &env, pool).filter(|v| v.is_finite())
}

/// `true` when `expr` simplifies to the integer zero.
fn is_zero_const(expr: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(simplify(expr, pool).value), ExprData::Integer(n) if n.0 == 0)
}

/// `a / b`, built and simplified.
fn quot(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    simplify(pool.mul(vec![a, pool.pow(b, pool.integer(-1_i32))]), pool).value
}

/// Split a product into its `var`-free part and its `var`-dependent factors.
///
/// A non-`Mul` expression counts as a single dependent factor with a unit
/// constant, so every matcher below can be written against one shape.
fn split_constant(expr: ExprId, var: ExprId, pool: &ExprPool) -> (ExprId, Vec<ExprId>) {
    let args = match pool.get(expr) {
        ExprData::Mul(args) => args,
        _ => {
            return if is_free_of(expr, var, pool) {
                (expr, Vec::new())
            } else {
                (pool.integer(1_i32), vec![expr])
            }
        }
    };
    let mut consts: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for a in args {
        if is_free_of(a, var, pool) {
            consts.push(a);
        } else {
            rest.push(a);
        }
    }
    let c = match consts.len() {
        0 => pool.integer(1_i32),
        1 => consts[0],
        _ => pool.mul(consts),
    };
    (c, rest)
}

/// `Some((a, b))` for a `Pow` factor `d^(-1)`, with `d = a·var + b` linear and
/// `a ≠ 0`.
fn reciprocal_linear(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let ExprData::Pow { base, exp } = pool.get(expr) else {
        return None;
    };
    if !matches!(pool.get(exp), ExprData::Integer(n) if n.0 == -1) {
        return None;
    }
    let (a, b) = is_linear_in(base, var, pool)?;
    (!is_zero_const(a, pool)).then_some((a, b))
}

// ---------------------------------------------------------------------------
// Matcher 1 — Ei / Si / Ci / Shi / Chi
// ---------------------------------------------------------------------------

/// `c · f(a·x+b) / (p·x+q)` with `f ∈ {exp, sin, cos, sinh, cosh}`.
///
/// The reduction is a change of variable, not a table lookup: when the
/// denominator is proportional to the argument (`a·q = b·p`), setting `u = g`
/// gives `∫f(u)/u du / p`, whose value is the corresponding integral function.
/// When it is not proportional there is no such reduction and the matcher
/// declines — except for `exp`, where `exp(a·x+b) = e^b·exp(a·x)` lets a pure
/// `q = 0` denominator through.
fn match_quotient_family(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (c, rest) = split_constant(expr, var, pool);
    if rest.len() != 2 {
        return None;
    }

    let mut numerator: Option<(&'static str, ExprId, ExprId, ExprId)> = None; // (F, g, a, b)
    let mut denominator: Option<(ExprId, ExprId)> = None; // (p, q)

    for &factor in &rest {
        if let ExprData::Func { name, args } = pool.get(factor) {
            if args.len() == 1 {
                if let Some(out) = special_integral_name(&name) {
                    let (a, b) = is_linear_in(args[0], var, pool)?;
                    if is_zero_const(a, pool) || numerator.is_some() {
                        return None;
                    }
                    numerator = Some((out, args[0], a, b));
                    continue;
                }
            }
        }
        if let Some((p, q)) = reciprocal_linear(factor, var, pool) {
            if denominator.is_some() {
                return None;
            }
            denominator = Some((p, q));
            continue;
        }
        return None;
    }

    let (out, g, a, b) = numerator?;
    let (p, q) = denominator?;

    // Proportional: `p·x+q = (p/a)·(a·x+b)`, so `∫f(g)/(p·x+q) dx = F(g)/p`.
    let cross = pool.add(vec![
        pool.mul(vec![a, q]),
        pool.mul(vec![pool.integer(-1_i32), b, p]),
    ]);
    if is_zero_const(cross, pool) {
        let fg = pool.func(out, vec![g]);
        return Some(pool.mul(vec![c, pool.pow(p, pool.integer(-1_i32)), fg]));
    }

    // `exp` only: `exp(a·x+b)/(p·x) = (e^b/p)·exp(a·x)/x`, and `∫exp(a·x)/x dx
    // = Ei(a·x)`.  The other four have no constant-factor split of this kind.
    if out == "Ei" && is_zero_const(q, pool) {
        let ax = simplify(pool.mul(vec![a, var]), pool).value;
        let eb = pool.func("exp", vec![b]);
        let ei = pool.func("Ei", vec![ax]);
        return Some(pool.mul(vec![c, eb, pool.pow(p, pool.integer(-1_i32)), ei]));
    }

    None
}

// ---------------------------------------------------------------------------
// Matcher 2 — li
// ---------------------------------------------------------------------------

/// `c / log(a·x+b)` → `(c/a)·li(a·x+b)`.
fn match_log_reciprocal(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (c, rest) = split_constant(expr, var, pool);
    if rest.len() != 1 {
        return None;
    }
    let ExprData::Pow { base, exp } = pool.get(rest[0]) else {
        return None;
    };
    if !matches!(pool.get(exp), ExprData::Integer(n) if n.0 == -1) {
        return None;
    }
    let ExprData::Func { name, args } = pool.get(base) else {
        return None;
    };
    if name != "log" || args.len() != 1 {
        return None;
    }
    // Only the slope matters: `li(a·x+b)` differentiates to `a/log(a·x+b)`,
    // whatever `b` is, so the intercept never reaches the answer.
    let (a, _b) = is_linear_in(args[0], var, pool)?;
    if is_zero_const(a, pool) {
        return None;
    }
    let li = pool.func("li", vec![args[0]]);
    Some(pool.mul(vec![c, pool.pow(a, pool.integer(-1_i32)), li]))
}

// ---------------------------------------------------------------------------
// Matcher 3 — erf
// ---------------------------------------------------------------------------

/// `c · exp(A·x² + B·x + C)` with `A < 0` → `erf`.
///
/// Completing the square gives `A·(x+h)² + K` with `h = B/(2A)` and
/// `K = C − A·h²`, and `∫exp(−α²w²) dw = (√π/2α)·erf(α·w)` with `α = √(−A)`.
///
/// `A > 0` is **not** covered: the answer is `erfi`, which is not a registered
/// primitive, and inventing it here would produce an expression the gate cannot
/// differentiate.
fn match_gaussian(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (c, rest) = split_constant(expr, var, pool);
    if rest.len() != 1 {
        return None;
    }
    let ExprData::Func { name, args } = pool.get(rest[0]) else {
        return None;
    };
    if name != "exp" || args.len() != 1 {
        return None;
    }
    let (aa, bb, cc) = quadratic_coeffs(args[0], var, pool)?;
    // `A ≥ 0` — including a NaN that got this far — is not this matcher's:
    // `A > 0` is `erfi`, which is not a registered primitive, and `A = 0` is
    // not a Gaussian at all.
    if aa >= 0.0 || !aa.is_finite() {
        return None;
    }
    let alpha = (-aa).sqrt();
    let h = bb / (2.0 * aa);
    let k = cc - aa * h * h;
    if !alpha.is_finite() || !h.is_finite() || !k.is_finite() {
        return None;
    }
    let coeff = k.exp() * std::f64::consts::PI.sqrt() / (2.0 * alpha);
    if !coeff.is_finite() {
        return None;
    }
    let shifted = if h == 0.0 {
        var
    } else {
        pool.add(vec![var, pool.float(h, 53)])
    };
    let arg = if alpha == 1.0 {
        shifted
    } else {
        pool.mul(vec![pool.float(alpha, 53), shifted])
    };
    let erf = pool.func("erf", vec![arg]);
    Some(pool.mul(vec![c, pool.float(coeff, 53), erf]))
}

/// `Some((A, B, C))` for `A·var² + B·var + C` with numeric coefficients and
/// `A ≠ 0`.  Anything else — a higher power, a non-constant coefficient, a
/// coefficient that will not evaluate — declines.
fn quadratic_coeffs(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(f64, f64, f64)> {
    // Work on the normalised form: `-x^2` parses as `Mul(-1, Pow(x, 2))` and
    // sums only become a flat `Add` after `simplify`.  A non-`Add` normal form
    // is a single term.
    let normalised = simplify(expr, pool).value;
    let terms = match pool.get(normalised) {
        ExprData::Add(args) => args,
        _ => vec![normalised],
    };
    let (mut a, mut b, mut c) = (0.0_f64, 0.0_f64, 0.0_f64);
    for t in terms {
        if is_free_of(t, var, pool) {
            c += const_f64(t, var, pool)?;
            continue;
        }
        let (coeff, degree) = monomial(t, var, pool)?;
        match degree {
            1 => b += coeff,
            2 => a += coeff,
            _ => return None,
        }
    }
    (a != 0.0).then_some((a, b, c))
}

/// `Some((coefficient, degree))` for `k·var^d` with `d ∈ {1, 2}`.
fn monomial(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(f64, u32)> {
    if expr == var {
        return Some((1.0, 1));
    }
    if let ExprData::Pow { base, exp } = pool.get(expr) {
        if base == var {
            let e = crate::kernel::try_expr_f64(exp, pool)?;
            if e == 1.0 || e == 2.0 {
                return Some((1.0, e as u32));
            }
        }
        return None;
    }
    let ExprData::Mul(args) = pool.get(expr) else {
        return None;
    };
    let mut coeff = 1.0_f64;
    let mut degree: Option<u32> = None;
    for a in args {
        if is_free_of(a, var, pool) {
            coeff *= const_f64(a, var, pool)?;
            continue;
        }
        let (k, d) = monomial(a, var, pool)?;
        coeff *= k;
        degree = Some(degree.unwrap_or(0) + d);
    }
    let d = degree?;
    (d == 1 || d == 2).then_some((coeff, d))
}

// ---------------------------------------------------------------------------
// Matcher 4 — Fresnel
// ---------------------------------------------------------------------------

/// `c · sin(A·x²)` → `c·sgn(A)·√(π/2|A|)·S(x·√(2|A|/π))`, and the `cos`/`C`
/// analogue.
///
/// **The scaling is not optional.**  The registered primitives use the DLMF
/// normalisation `S(x) = ∫₀ˣ sin(πt²/2) dt`, so `∫sin(x²) dx` is
/// `√(π/2)·S(x·√(2/π))` and *not* `S(x)`.  Emitting the unscaled form would be
/// a wrong answer that a string-matching test would happily accept; the gate
/// below rejects it, which is why the gate is not optional either.
///
/// `sin` is odd and `cos` even, which is where the `sgn(A)` comes from.
fn match_fresnel(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (c, rest) = split_constant(expr, var, pool);
    if rest.len() != 1 {
        return None;
    }
    let ExprData::Func { name, args } = pool.get(rest[0]) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let (out, odd) = match name.as_str() {
        "sin" => ("fresnels", true),
        "cos" => ("fresnelc", false),
        _ => return None,
    };
    // A pure quadratic only: `sin(A·x² + B·x)` needs an angle-addition split
    // this module does not do, and `B ≠ 0` would silently give a wrong answer.
    let (aa, bb, cc) = quadratic_coeffs(args[0], var, pool)?;
    if bb != 0.0 || cc != 0.0 {
        return None;
    }
    let mag = aa.abs();
    let scale = (2.0 * mag / std::f64::consts::PI).sqrt();
    let outer = (std::f64::consts::PI / (2.0 * mag)).sqrt();
    if !scale.is_finite() || !outer.is_finite() || scale == 0.0 {
        return None;
    }
    let sign = if odd && aa < 0.0 { -1.0 } else { 1.0 };
    let arg = pool.mul(vec![pool.float(scale, 53), var]);
    let f = pool.func(out, vec![arg]);
    Some(pool.mul(vec![c, pool.float(sign * outer, 53), f]))
}

// ---------------------------------------------------------------------------
// Matcher 5 — Li₂
// ---------------------------------------------------------------------------

/// `c · log(x) / (a + b·x)` with `a·b ≠ 0` →
/// `(c/b)·[log(x)·log(1+m·x) + Li₂(−m·x)]`, `m = b/a`.
///
/// Note the answer needs `log(1 + m·x)`, **not** `log(a + b·x)`: they differ by
/// `log(a)·log(x)`, which is not a constant of integration.
fn match_dilog(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (c, rest) = split_constant(expr, var, pool);
    if rest.len() != 2 {
        return None;
    }
    let mut has_log = false;
    let mut denom: Option<(ExprId, ExprId)> = None;
    for &factor in &rest {
        if let ExprData::Func { name, args } = pool.get(factor) {
            if name == "log" && args.len() == 1 && args[0] == var && !has_log {
                has_log = true;
                continue;
            }
        }
        if let Some((b, a)) = reciprocal_linear(factor, var, pool) {
            if denom.is_some() || is_zero_const(a, pool) {
                return None;
            }
            denom = Some((a, b));
            continue;
        }
        return None;
    }
    if !has_log {
        return None;
    }
    let (a, b) = denom?;
    let m = quot(b, a, pool);
    let mx = simplify(pool.mul(vec![m, var]), pool).value;
    let one_plus = pool.add(vec![pool.integer(1_i32), mx]);
    let logx = pool.func("log", vec![var]);
    let term1 = pool.mul(vec![logx, pool.func("log", vec![one_plus])]);
    let neg_mx = pool.mul(vec![pool.integer(-1_i32), mx]);
    let term2 = pool.func("dilog", vec![neg_mx]);
    let inner = pool.add(vec![term1, term2]);
    Some(pool.mul(vec![c, pool.pow(b, pool.integer(-1_i32)), inner]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn setup() -> (ExprPool, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        (pool, x)
    }

    /// `∫eˣ/x dx = Ei(x)`.
    #[test]
    fn exp_over_x_is_ei() {
        let (pool, x) = setup();
        let f = pool.mul(vec![
            pool.func("exp", vec![x]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        let (out, _) = try_special_antiderivative(f, x, &pool).expect("Ei");
        assert_eq!(basis_functions_used(out, &pool), vec!["Ei"]);
    }

    /// `∫sin(x)/x dx = Si(x)`, `∫cos(x)/x dx = Ci(x)`.
    #[test]
    fn sinc_and_cosc() {
        for (fname, expected) in [("sin", "Si"), ("cos", "Ci")] {
            let (pool, x) = setup();
            let f = pool.mul(vec![
                pool.func(fname, vec![x]),
                pool.pow(x, pool.integer(-1_i32)),
            ]);
            let (out, _) = try_special_antiderivative(f, x, &pool).expect("special");
            assert_eq!(basis_functions_used(out, &pool), vec![expected]);
        }
    }

    /// `∫dx/log(x) = li(x)`.
    #[test]
    fn recip_log_is_li() {
        let (pool, x) = setup();
        let f = pool.pow(pool.func("log", vec![x]), pool.integer(-1_i32));
        let (out, _) = try_special_antiderivative(f, x, &pool).expect("li");
        assert_eq!(basis_functions_used(out, &pool), vec!["li"]);
    }

    /// `∫exp(−x²) dx = (√π/2)·erf(x)`.
    #[test]
    fn gaussian_is_erf() {
        let (pool, x) = setup();
        let arg = pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))]);
        let f = pool.func("exp", vec![arg]);
        let (out, _) = try_special_antiderivative(f, x, &pool).expect("erf");
        assert_eq!(basis_functions_used(out, &pool), vec!["erf"]);
    }

    /// `∫exp(+x²) dx` needs `erfi`, which is not registered: decline, do not
    /// invent a name the gate cannot differentiate.
    #[test]
    fn positive_gaussian_declines() {
        let (pool, x) = setup();
        let f = pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))]);
        assert!(try_special_antiderivative(f, x, &pool).is_none());
    }

    /// The Fresnel scaling is the whole content of the reduction: the unscaled
    /// `S(x)` is a wrong answer and the gate must reject it.
    #[test]
    fn fresnel_is_scaled() {
        let (pool, x) = setup();
        let f = pool.func("sin", vec![pool.pow(x, pool.integer(2_i32))]);
        let (out, _) = try_special_antiderivative(f, x, &pool).expect("fresnel");
        assert_eq!(basis_functions_used(out, &pool), vec!["fresnels"]);

        let unscaled = pool.func("fresnels", vec![x]);
        assert!(
            verify_antiderivative_status(unscaled, f, x, &pool).is_none(),
            "unscaled S(x) must not verify against sin(x²)"
        );
    }

    /// `∫log(x)/(1+x) dx = log(x)·log(1+x) + Li₂(−x)`.
    #[test]
    fn log_over_one_plus_x_is_dilog() {
        let (pool, x) = setup();
        let den = pool.add(vec![pool.integer(1_i32), x]);
        let f = pool.mul(vec![
            pool.func("log", vec![x]),
            pool.pow(den, pool.integer(-1_i32)),
        ]);
        let (out, _) = try_special_antiderivative(f, x, &pool).expect("dilog");
        assert_eq!(basis_functions_used(out, &pool), vec!["dilog"]);
    }

    /// An elementary integrand must not be captured: `∫2x·exp(−x²) dx` is
    /// `−exp(−x²)` and has no business becoming an `erf`.
    #[test]
    fn elementary_gaussian_multiple_declines() {
        let (pool, x) = setup();
        let arg = pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))]);
        let f = pool.mul(vec![pool.integer(2_i32), x, pool.func("exp", vec![arg])]);
        assert!(try_special_antiderivative(f, x, &pool).is_none());
    }

    /// The module cannot express a `NonElementary` verdict — the return type
    /// has no room for one.  This is the same pin `by_parts` and `norman` carry.
    #[test]
    fn decline_cannot_become_non_elementary() {
        let (pool, x) = setup();
        let f = pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))]);
        assert!(try_special_antiderivative(f, x, &pool).is_none());
        assert!(matches!(
            decline(f, &pool),
            IntegrationError::NotImplemented(_)
        ));
    }

    /// `classify` reports vocabulary, and reports nothing for an elementary
    /// answer.
    #[test]
    fn classify_splits_the_two_cases() {
        let (pool, x) = setup();
        let elementary = DerivedExpr::new(pool.pow(x, pool.integer(2_i32)));
        assert!(matches!(
            classify(elementary, &pool),
            IntegrationAnswer::Elementary(_)
        ));
        let special = DerivedExpr::new(pool.func("Si", vec![x]));
        let answer = classify(special, &pool);
        assert!(answer.is_non_elementary_closed_form());
        assert_eq!(answer.basis(), ["Si"]);
    }

    /// `EllipticPi` is not in the basis: it differentiates but cannot be
    /// bounded by the validated tier, so this module will never introduce it.
    #[test]
    fn elliptic_pi_is_not_in_the_basis() {
        assert!(!SPECIAL_BASIS.contains(&"EllipticPi"));
    }
}
