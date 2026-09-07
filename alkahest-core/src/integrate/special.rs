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
//! > is not a theorem.  Re-reading the existing certificate as the stronger
//! > statement would manufacture a false one, which is exactly the defect eight
//! > families of this codebase were already found to have.
//!
//! The witness for that has always been an integrand that is non-elementary
//! (so the certificate is sound) *and* expressible over the basis (so the
//! stronger reading would be false) *and* not in the table.  It used to be
//! `∫sin(x)/x² dx`; `match_quotient_power` now answers that one
//! (`−sin(x)/x + Ci(x)`), so two measured replacements are `∫exp(−x²)/x dx`
//! (certified `E-INT-004` by the Risch tower; equal to `Ei(−x²)/2`, which needs
//! a *quadratic* argument the table has no entry for) and
//! `∫sin(2x+3)/(x+1) dx` (certified `E-INT-004` by `known_nonelementary`; equal
//! to `cos(1)·Si(2x+2) + sin(1)·Ci(2x+2)`, which needs an angle-addition split
//! the table has no entry for).  **Closing table entries never shrinks the gap
//! between the two statements**; it only moves which example demonstrates it,
//! which is precisely why the certificate must not be read as tracking the
//! table.
//!
//! So [`IntegrationError::NonElementary`] keeps precisely the meaning and the
//! wording it has always had: *no **elementary** antiderivative exists*.  The
//! strengthening is available to whoever builds a decision procedure for the
//! extended basis, and this module's docs are the place to record that it is not
//! built.
//!
//! # Soundness
//!
//! Every candidate goes through
//! [`verify_antiderivative_status_parametric`] — symbolic `d/dx F − f ≡ 0`
//! first, then the in-domain `f64` screen, swept over the declared domain of
//! any free parameter — before it is returned, exactly as `elliptic_output`,
//! `by_parts` and `norman` do.  A candidate that cannot be confirmed is
//! discarded and the caller falls through to whatever verdict it already had.
//! There is **no path from this module to `NonElementary`**: it returns
//! `Some(F)` or `None`, and `None` is a decline.
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
//! | `c·f(g)/dⁿ`, same heads, `n ≥ 2` | the same, plus elementary terms |
//! | `c·exp(a·x+b)/(p·x)ⁿ` | `Ei` (the `e^b` is pulled out) |
//! | `c/log(g)`, `g` linear | `li` |
//! | `c·exp(A·x²+B·x+C)`, `A < 0` numeric | `erf` |
//! | `c·exp(A·x²+B·x+C)`, `A` symbolic and provably negative | `erf` |
//! | `c·xⁿ·exp(A·x²+C)`, `n` even, `A < 0` | `erf` plus elementary terms |
//! | `c·sin(A·x²)` / `c·cos(A·x²)` | Fresnel `S` / `C` |
//! | `c·log(x)/(a+b·x)`, `a·b ≠ 0` | `Li₂` |
//!
//! Not covered, and honest about it: `A > 0` in the Gaussian (would need
//! `erfi`, which is not a registered primitive); an `A` whose sign is not
//! decidable from the symbol's declared domain (`∫exp(−a·x²) dx` for a plain
//! `Domain::Real` `a` — the answer would be `NaN` for every `a < 0`, so it is
//! refused rather than emitted, and declaring `a` positive is what makes it
//! answerable); `xⁿ` against a *shifted* Gaussian (`B ≠ 0`), which needs a
//! binomial expansion after the substitution; and non-linear arguments other
//! than the pure quadratics above.

use std::collections::HashMap;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::{simplify, simplify_expanded};

use super::engine::{
    is_free_of, is_linear_in, special_integral_name, verify_antiderivative_status_parametric,
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
        match_quotient_power(work, var, pool),
        match_log_reciprocal(work, var, pool),
        match_gaussian(work, var, pool),
        match_gaussian_symbolic(work, var, pool),
        match_gaussian_moment(work, var, pool),
        match_fresnel(work, var, pool),
        match_dilog(work, var, pool),
    ];

    for candidate in candidates.into_iter().flatten() {
        let f = simplify(candidate, pool).value;
        // The parametric gate is the ordinary gate whenever the integrand
        // carries no free parameter, so nothing that verified before verifies
        // differently now; the extra reach is exactly the `∫exp(−a·x²) dx`
        // family, whose `a` the plain `var`-only grid cannot bind.
        if let Some(evidence) = verify_antiderivative_status_parametric(f, expr, var, pool) {
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
// Matcher 1b — Ei/Si/Ci/Shi/Chi under a higher denominator power
// ---------------------------------------------------------------------------

/// `c · f(a·x+b) / (p·x+q)ⁿ` with `n ≥ 2` — the same five heads, reduced by
/// parts down to [`match_quotient_family`]'s case.
///
/// `∫sin(x)/x² dx` is the example this module's own docs have carried since it
/// was written, as the standing proof that `NonElementary` must not be re-read
/// as "not expressible over the basis either": it is non-elementary, it *is*
/// `−sin(x)/x + Ci(x)`, and no matcher here found it.  That gap is now closed —
/// which does not change the argument, only removes this particular witness for
/// it.  The certificate still means exactly what it always meant; the module
/// docs name the two integrands that carry the witness now.
///
/// The reduction is integration by parts with `u = f(g)`, `dv = g⁻ⁿ dg`:
///
/// ```text
///   Jₙ(u) = ∫f(u)/uⁿ du = −f(u)/((n−1)·uⁿ⁻¹) + (σ/(n−1))·Jₙ₋₁(u)
///   J₁(u) = F(u)                       F ∈ {Ei, Si, Ci, Shi, Chi}
/// ```
///
/// where `f′ = σ·f_next` — `σ = −1` for `cos` (whose derivative is `−sin`) and
/// `+1` for the other four.  With `d = (p/a)·g` and `dg = a·dx`, the change of
/// variable contributes `aⁿ⁻¹/pⁿ`.
///
/// Every emission still passes the same `d/dx F = f` gate as the `n = 1` case,
/// so a slipped sign in the recursion is caught rather than shipped.
fn match_quotient_power(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (c, rest) = split_constant(expr, var, pool);
    if rest.len() != 2 {
        return None;
    }

    let mut numerator: Option<(String, ExprId, ExprId, ExprId)> = None; // (f, g, a, b)
    let mut denominator: Option<(ExprId, ExprId, u32)> = None; // (p, q, n)

    for &factor in &rest {
        if let ExprData::Func { name, args } = pool.get(factor) {
            if args.len() == 1 && special_integral_name(&name).is_some() {
                let (a, b) = is_linear_in(args[0], var, pool)?;
                if is_zero_const(a, pool) || numerator.is_some() {
                    return None;
                }
                numerator = Some((name.clone(), args[0], a, b));
                continue;
            }
        }
        if let Some((p, q, n)) = reciprocal_linear_power(factor, var, pool) {
            if denominator.is_some() || n < 2 {
                return None;
            }
            denominator = Some((p, q, n));
            continue;
        }
        return None;
    }

    let (fname, g, a, b) = numerator?;
    let (p, q, n) = denominator?;

    // `∫f(g)/dⁿ dx` needs `d ∝ g`, exactly as in the `n = 1` case; the one
    // exception is again `exp`, whose `e^b` factors out of a `q = 0`
    // denominator.
    let cross = pool.add(vec![
        pool.mul(vec![a, q]),
        pool.mul(vec![pool.integer(-1_i32), b, p]),
    ]);
    let (arg, extra) = if is_zero_const(cross, pool) {
        (g, pool.integer(1_i32))
    } else if fname == "exp" && is_zero_const(q, pool) {
        (
            simplify(pool.mul(vec![a, var]), pool).value,
            pool.func("exp", vec![b]),
        )
    } else {
        return None;
    };

    let jacobian = pool.mul(vec![
        pool.pow(a, pool.integer(n as i32 - 1)),
        pool.pow(p, pool.integer(-(n as i32))),
    ]);
    let body = quotient_power_reduce(&fname, arg, n, pool)?;
    Some(pool.mul(vec![c, extra, jacobian, body]))
}

/// `Jₙ(u) = ∫f(u)/uⁿ du` of [`match_quotient_power`], built as an expression in
/// the already-substituted argument `u`.
fn quotient_power_reduce(fname: &str, u: ExprId, n: u32, pool: &ExprPool) -> Option<ExprId> {
    if n == 1 {
        return Some(pool.func(special_integral_name(fname)?, vec![u]));
    }
    // `f′ = σ·next`; `cos′ = −sin` is the only sign in the family.
    let (next, sigma) = match fname {
        "exp" => ("exp", 1_i32),
        "sin" => ("cos", 1),
        "cos" => ("sin", -1),
        "sinh" => ("cosh", 1),
        "cosh" => ("sinh", 1),
        _ => return None,
    };
    let nm1 = pool.integer(n as i32 - 1);
    let inv_nm1 = pool.pow(nm1, pool.integer(-1_i32));
    let head = pool.mul(vec![
        pool.integer(-1_i32),
        inv_nm1,
        pool.func(fname, vec![u]),
        pool.pow(u, pool.integer(-(n as i32 - 1))),
    ]);
    let tail = pool.mul(vec![
        pool.integer(sigma),
        inv_nm1,
        quotient_power_reduce(next, u, n - 1, pool)?,
    ]);
    Some(pool.add(vec![head, tail]))
}

/// `Some((a, b, n))` for a factor `(a·var + b)^(−n)` with `n ≥ 1` and `a ≠ 0`.
fn reciprocal_linear_power(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId, u32)> {
    let ExprData::Pow { base, exp } = pool.get(expr) else {
        return None;
    };
    let ExprData::Integer(k) = pool.get(exp) else {
        return None;
    };
    let k = i64::try_from(&k.0).ok()?;
    if k >= 0 {
        return None;
    }
    let n = u32::try_from(-k).ok()?;
    let (a, b) = is_linear_in(base, var, pool)?;
    (!is_zero_const(a, pool)).then_some((a, b, n))
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
// Matcher 3b — erf with symbolic coefficients
// ---------------------------------------------------------------------------

/// `c · exp(A·x² + B·x + C)` with **symbolic** `A`, `B`, `C` and `A` provably
/// negative → `erf`.
///
/// [`match_gaussian`] evaluates the three coefficients to `f64`, which is what
/// makes `∫exp(−a·x²) dx` and `∫exp(−(x−b)²) dx` invisible to it: `a` and `b`
/// have no `f64` value.  The reduction is identical — complete the square,
/// substitute `u = α·(x + h)` with `α = √(−A)` — but carried out on
/// expressions:
///
/// ```text
///   A·x² + B·x + C = A·(x + h)² + K,   h = B/(2A),   K = C − A·h²
///   ∫exp(−α²·u²) du = (√π / 2α)·erf(α·u)
/// ```
///
/// # Why `A` must be *provably* negative, not just not-numerically-positive
///
/// `√(−A)` is a real number only when `A < 0`, and for `A > 0` the honest
/// answer is `erfi`, which is not a registered primitive.  Emitting the `erf`
/// form for an `A` of unknown sign would hand back an expression that is `NaN`
/// over half its parameter range — the "domain hole" shape 3.9.0 made a
/// refusal.  So the sign has to come from the *declaration*: a negative
/// literal, or `−1` times a [`crate::kernel::Domain::Positive`] symbol.  An
/// `a` declared merely `Real` declines here, and declines again at the gate
/// (which samples it on both sides of zero) if it somehow got this far.
///
/// The `√π/2` is emitted as a `f64` literal rather than as `sqrt(pi)` on
/// purpose: `pi` is an ordinary free symbol in this codebase, and an answer
/// carrying one is an answer [`crate::jit::eval_interp`] cannot evaluate — the
/// numeric half of the gate would go blind, and so would every caller that
/// later asks for a number.
fn match_gaussian_symbolic(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
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
    let (aa, bb, cc) = quadratic_coeffs_sym(args[0], var, pool)?;
    let alpha = negated_square_root(aa, pool)?;

    // h = B/(2A); K = C − A·h².
    let two_a = simplify(pool.mul(vec![pool.integer(2_i32), aa]), pool).value;
    let h = quot(bb, two_a, pool);
    let shifted = simplify(pool.add(vec![var, h]), pool).value;
    let h2 = pool.pow(h, pool.integer(2_i32));
    let k = simplify(
        pool.add(vec![cc, pool.mul(vec![pool.integer(-1_i32), aa, h2])]),
        pool,
    )
    .value;

    let arg = simplify(pool.mul(vec![alpha, shifted]), pool).value;
    let erf = pool.func("erf", vec![arg]);
    let outer = pool.mul(vec![
        c,
        pool.func("exp", vec![k]),
        pool.float(std::f64::consts::PI.sqrt() / 2.0, 53),
        pool.pow(alpha, pool.integer(-1_i32)),
        erf,
    ]);
    Some(outer)
}

// ---------------------------------------------------------------------------
// Matcher 3c — Gaussian moments `∫xⁿ·exp(A·x² + C) dx`
// ---------------------------------------------------------------------------

/// `c · xⁿ · exp(A·x² + C)` with `A` provably negative and `n` a non-negative
/// integer → `erf` plus elementary terms.
///
/// Integration by parts against `d(G) = 2A·x·G dx`, `G = exp(A·x² + C)`:
///
/// ```text
///   ∫xⁿ·G dx = xⁿ⁻¹·G/(2A) − ((n−1)/(2A))·∫xⁿ⁻²·G dx
///   ∫x·G  dx = G/(2A)
///   ∫G    dx = exp(C)·(√π/2α)·erf(α·x),   α = √(−A)
/// ```
///
/// The recursion drops `n` by two, so an **odd** `n` bottoms out at `∫x·G` and
/// the answer is elementary, while an **even** `n` bottoms out at `∫G` and
/// keeps exactly one `erf`.  That is the reason `∫x³·exp(−x²) dx` was already
/// answered by the elementary pipeline and `∫x²·exp(−x²) dx` was an
/// `E-INT-004`: the Risch tower is right that no elementary antiderivative
/// exists for the even case, and wrong only about what follows from that.
///
/// **Odd `n` is excluded**, even though the recursion handles it.  An odd `n`
/// bottoms out at `∫x·G` and produces an answer with no special function in it
/// at all, and this module's contract is that it emits *non-elementary* closed
/// forms — a second, weaker route to an elementary answer the rule engine
/// already has is not an improvement, and `elementary_gaussian_multiple_declines`
/// exists to keep it that way.  The base case `n = 1` stays in
/// [`gaussian_moment`] because the even recursion never reaches it.
///
/// A linear term (`B ≠ 0`) is **not** covered.  The shift `u = x + B/(2A)`
/// turns `xⁿ` into `(u − h)ⁿ`, which needs a binomial expansion this matcher
/// does not do; declining is a decline, not a claim.
fn match_gaussian_moment(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (c, rest) = split_constant(expr, var, pool);
    if rest.len() != 2 {
        return None;
    }

    let mut power: Option<u32> = None;
    let mut gaussian: Option<(ExprId, ExprId)> = None; // (G, A)
    for &factor in &rest {
        if let Some(n) = var_power(factor, var, pool) {
            if power.is_some() {
                return None;
            }
            power = Some(n);
            continue;
        }
        if let ExprData::Func { name, args } = pool.get(factor) {
            if name == "exp" && args.len() == 1 && gaussian.is_none() {
                let (aa, bb, _cc) = quadratic_coeffs_sym(args[0], var, pool)?;
                if !is_zero_const(bb, pool) {
                    return None;
                }
                gaussian = Some((factor, aa));
                continue;
            }
        }
        return None;
    }

    let n = power?;
    let (g, aa) = gaussian?;
    // `n = 0` is `match_gaussian`'s / `match_gaussian_symbolic`'s case; odd `n`
    // is elementary and not this module's business — see the doc comment.
    if n < 2 || n % 2 != 0 {
        return None;
    }
    let alpha = negated_square_root(aa, pool)?;
    Some(pool.mul(vec![c, gaussian_moment(n, g, aa, alpha, var, pool)?]))
}

/// The recursion of [`match_gaussian_moment`], as an expression.
fn gaussian_moment(
    n: u32,
    g: ExprId,
    aa: ExprId,
    alpha: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    // `2A` is non-zero: `quadratic_coeffs_sym` rejects a vanishing leading
    // coefficient and `negated_square_root` has already proved `−A > 0`.
    let inv_two_a = pool.pow(
        pool.mul(vec![pool.integer(2_i32), aa]),
        pool.integer(-1_i32),
    );
    match n {
        // ∫exp(A·x² + C) dx.  `G` carries the `exp(C)` factor itself, so the
        // constant is never re-derived: `G/exp(A·x²)` is `exp(C)`.
        0 => {
            let axx = pool.mul(vec![aa, pool.pow(var, pool.integer(2_i32))]);
            let expc = pool.mul(vec![
                g,
                pool.pow(pool.func("exp", vec![axx]), pool.integer(-1_i32)),
            ]);
            let arg = simplify(pool.mul(vec![alpha, var]), pool).value;
            Some(pool.mul(vec![
                simplify(expc, pool).value,
                pool.float(std::f64::consts::PI.sqrt() / 2.0, 53),
                pool.pow(alpha, pool.integer(-1_i32)),
                pool.func("erf", vec![arg]),
            ]))
        }
        1 => Some(pool.mul(vec![g, inv_two_a])),
        _ => {
            let head = pool.mul(vec![
                pool.pow(var, pool.integer(n as i32 - 1)),
                g,
                inv_two_a,
            ]);
            let tail = pool.mul(vec![
                pool.integer(-(n as i32 - 1)),
                inv_two_a,
                gaussian_moment(n - 2, g, aa, alpha, var, pool)?,
            ]);
            Some(pool.add(vec![head, tail]))
        }
    }
}

/// `Some(n)` when `factor` is `var^n` with `n` a non-negative integer literal.
fn var_power(factor: ExprId, var: ExprId, pool: &ExprPool) -> Option<u32> {
    if factor == var {
        return Some(1);
    }
    let ExprData::Pow { base, exp } = pool.get(factor) else {
        return None;
    };
    if base != var {
        return None;
    }
    let ExprData::Integer(n) = pool.get(exp) else {
        return None;
    };
    u32::try_from(n.0).ok()
}

/// `Some((A, B, C))` for `A·var² + B·var + C` as **expressions**, with `A` a
/// non-zero `var`-free coefficient.
///
/// The `f64` twin, `quadratic_coeffs`, is kept: it is what
/// [`match_gaussian`]'s float-folded output is built from, and re-routing that
/// through here would change the spelling of answers that already verify.
fn quadratic_coeffs_sym(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId, ExprId)> {
    // Expansion, not plain `simplify`: the whole point of this matcher is the
    // shifted Gaussian, and `exp(−(x−b)²)` keeps its exponent as
    // `−1·(x + −b)²` — a `Pow` whose base is an `Add`, which `monomial_sym`
    // (correctly) has no reading of.  `simplify_expanded` multiplies it out to
    // `−x² + 2·b·x − b²`, which is the quadratic this function is looking for.
    let normalised = simplify_expanded(expr, pool).value;
    let terms = match pool.get(normalised) {
        ExprData::Add(args) => args,
        _ => vec![normalised],
    };
    let mut buckets: [Vec<ExprId>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for t in terms {
        let (coeff, degree) = monomial_sym(t, var, pool)?;
        buckets[degree as usize].push(coeff);
    }
    let collapse = |v: Vec<ExprId>| match v.len() {
        0 => pool.integer(0_i32),
        1 => v[0],
        _ => pool.add(v),
    };
    let [c0, c1, c2] = buckets;
    let aa = simplify(collapse(c2), pool).value;
    if is_zero_const(aa, pool) {
        return None;
    }
    Some((
        aa,
        simplify(collapse(c1), pool).value,
        simplify(collapse(c0), pool).value,
    ))
}

/// `Some((coefficient, degree))` for one term of a quadratic in `var`, with the
/// coefficient left symbolic.  Degrees above two decline.
fn monomial_sym(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, u32)> {
    if is_free_of(expr, var, pool) {
        return Some((expr, 0));
    }
    if expr == var {
        return Some((pool.integer(1_i32), 1));
    }
    if let ExprData::Pow { base, exp } = pool.get(expr) {
        if base != var {
            return None;
        }
        let ExprData::Integer(n) = pool.get(exp) else {
            return None;
        };
        let d = u32::try_from(n.0).ok()?;
        return (d <= 2).then_some((pool.integer(1_i32), d));
    }
    let ExprData::Mul(args) = pool.get(expr) else {
        return None;
    };
    let mut coeffs: Vec<ExprId> = Vec::new();
    let mut degree = 0_u32;
    for a in args {
        let (k, d) = monomial_sym(a, var, pool)?;
        coeffs.push(k);
        degree += d;
    }
    if degree > 2 {
        return None;
    }
    let coeff = match coeffs.len() {
        1 => coeffs[0],
        _ => pool.mul(coeffs),
    };
    Some((coeff, degree))
}

/// `Some(√(−A))` when `−A` is **provably** a positive real, else `None`.
///
/// "Provably" here is structural and deliberately small: a positive numeric
/// literal, a symbol declared [`crate::kernel::Domain::Positive`], an `exp`, an
/// even power, or a product/sum built from those.  It is a sufficient
/// condition, never a necessary one — declining is free, and a wrong `yes`
/// would emit `√` of a negative number.
fn negated_square_root(aa: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let neg = simplify(pool.mul(vec![pool.integer(-1_i32), aa]), pool).value;
    if !is_structurally_positive(neg, pool) {
        return None;
    }
    Some(simplify(pool.func("sqrt", vec![neg]), pool).value)
}

/// A sufficient structural test for `e > 0`.
fn is_structurally_positive(e: ExprId, pool: &ExprPool) -> bool {
    if let Some(v) = crate::kernel::try_expr_f64(e, pool) {
        return v > 0.0;
    }
    match pool.get(e) {
        ExprData::Symbol { domain, .. } => domain == crate::kernel::Domain::Positive,
        ExprData::Func { name, args } => {
            name == "exp"
                || (name == "sqrt" && args.len() == 1 && is_structurally_positive(args[0], pool))
        }
        ExprData::Mul(args) | ExprData::Add(args) => {
            args.iter().all(|&a| is_structurally_positive(a, pool))
        }
        // A positive base raised to any real power is positive.  An **even
        // power of an unsigned base is deliberately not accepted**, even though
        // `b² ≥ 0`: the `≥` is the problem.  `∫exp(−b²·x²) dx` would emit
        // `√π/(2b)·erf(b·x)`, which is right for every `b ≠ 0` and undefined at
        // `b = 0` — where the integrand is the perfectly ordinary `1`.  The
        // parameter sweep would never sample exactly `0` and so would never
        // catch it, and "correct except at one parameter value, silently" is
        // the shape of answer this codebase refuses.  A `b` the caller has
        // declared `NonZero` or `Positive` is a different question and is
        // accepted by the `Symbol` arm above.
        ExprData::Pow { base, .. } => is_structurally_positive(base, pool),
        _ => false,
    }
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
    use crate::integrate::engine::verify_antiderivative_status;
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

    /// Parse a source string against a fresh pool carrying `x` real, `p`
    /// positive and `b` real — the three declarations the parameter cases need.
    fn parsed(src: &str) -> (ExprPool, ExprId, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = std::collections::HashMap::from([
            ("x".to_owned(), x),
            ("b".to_owned(), pool.symbol("b", Domain::Real)),
            ("a".to_owned(), pool.symbol("a", Domain::Real)),
            ("p".to_owned(), pool.symbol("p", Domain::Positive)),
            ("q".to_owned(), pool.symbol("q", Domain::Positive)),
        ]);
        let f = crate::parse(src, &pool, &mut syms).expect("parse");
        (pool, f, x)
    }

    /// `∫exp(−p·x²) dx = (√π/2√p)·erf(√p·x)` for a symbol *declared* positive.
    /// The `f64` coefficient route cannot see this at all — `p` has no value.
    #[test]
    fn symbolic_gaussian_needs_only_a_positive_declaration() {
        let (pool, f, x) = parsed("exp(-p*x^2)");
        let (out, evidence) = try_special_antiderivative(f, x, &pool).expect("erf");
        assert_eq!(basis_functions_used(out, &pool), vec!["erf"]);
        assert_eq!(evidence, AntiderivativeVerification::Numeric);
    }

    /// The same integrand with an **undeclared-sign** `a` is refused, and that
    /// is the point: `√π/(2√a)·erf(√a·x)` is not a real number for `a < 0`, so
    /// emitting it would hand back an expression that is `NaN` over half its
    /// parameter range — the domain-hole shape the gate exists to catch.
    #[test]
    fn gaussian_with_an_unsigned_parameter_is_refused() {
        let (pool, f, x) = parsed("exp(-a*x^2)");
        assert!(try_special_antiderivative(f, x, &pool).is_none());
    }

    /// A linear shift is a substitution, not a new function: `∫exp(−(x−b)²) dx`
    /// is `(√π/2)·erf(x−b)` for every real `b`.
    #[test]
    fn shifted_gaussian_carries_its_parameter() {
        let (pool, f, x) = parsed("exp(-(x-b)^2)");
        let (out, _) = try_special_antiderivative(f, x, &pool).expect("erf");
        assert_eq!(basis_functions_used(out, &pool), vec!["erf"]);
    }

    /// `∫x²·exp(−x²) dx = −x·exp(−x²)/2 + (√π/4)·erf(x)` — one `erf`, one
    /// elementary term, from the by-parts recursion.
    #[test]
    fn even_gaussian_moment_is_erf_plus_elementary() {
        let (pool, f, x) = parsed("x^2*exp(-x^2)");
        let (out, _) = try_special_antiderivative(f, x, &pool).expect("erf");
        assert_eq!(basis_functions_used(out, &pool), vec!["erf"]);
    }

    /// `∫sin(x)/x² dx = −sin(x)/x + Ci(x)`.  This is the integrand this
    /// module's docs carried for as long as it was unreachable.
    #[test]
    fn sine_over_x_squared_is_ci_plus_elementary() {
        let (pool, f, x) = parsed("sin(x)/x^2");
        let (out, _) = try_special_antiderivative(f, x, &pool).expect("Ci");
        assert_eq!(basis_functions_used(out, &pool), vec!["Ci"]);
    }

    /// The `cos′ = −sin` sign is the one place the by-parts recursion can go
    /// wrong quietly, so the `cos` chain is pinned separately from the `sin`
    /// one: `∫cos(x)/x³ dx` reaches `Ci` through *two* reductions.
    #[test]
    fn cosine_over_x_cubed_keeps_its_sign() {
        let (pool, f, x) = parsed("cos(x)/x^3");
        let (out, _) = try_special_antiderivative(f, x, &pool).expect("Ci");
        assert_eq!(basis_functions_used(out, &pool), vec!["Ci"]);
    }

    /// A denominator that is **not** proportional to the argument has no
    /// change of variable to make, and the angle-addition split that would
    /// close `∫sin(2x+3)/(x+1) dx` is not in this module.  Decline, do not
    /// guess.
    #[test]
    fn non_proportional_denominator_declines() {
        let (pool, f, x) = parsed("sin(2*x+3)/(x+1)^2");
        assert!(try_special_antiderivative(f, x, &pool).is_none());
    }

    /// `−A = b²` is non-negative, not positive, and the difference is not
    /// pedantic: `√π/(2b)·erf(b·x)` is a fine antiderivative of `exp(−b²·x²)`
    /// for every `b ≠ 0` and is `0/0` at `b = 0`, where the integrand is the
    /// ordinary constant `1`.  The parameter sweep sails past `0` without ever
    /// landing on it, so the gate cannot be the thing that catches this —
    /// `is_structurally_positive` has to refuse the even power itself.
    #[test]
    fn an_even_power_is_not_a_positive_coefficient() {
        let (pool, f, x) = parsed("exp(-b^2*x^2)");
        assert!(try_special_antiderivative(f, x, &pool).is_none());
    }

    /// The same integrand with `b` declared positive is answerable, which is
    /// what makes the refusal above a statement about `b = 0` rather than about
    /// even powers.
    #[test]
    fn an_even_power_of_a_positive_symbol_is_a_positive_coefficient() {
        let (pool, f, x) = parsed("exp(-q^2*x^2)");
        let (out, _) = try_special_antiderivative(f, x, &pool).expect("erf");
        assert_eq!(basis_functions_used(out, &pool), vec!["erf"]);
    }

    /// `∫x³·exp(−x²) dx` is elementary and the rule engine already answers it;
    /// this module emits *non-elementary* closed forms and stays out of the
    /// way.  (The odd recursion would produce a correct answer — the point is
    /// that it must not be this module's to give.)
    #[test]
    fn odd_gaussian_moment_is_not_this_modules_business() {
        let (pool, f, x) = parsed("x^3*exp(-x^2)");
        assert!(try_special_antiderivative(f, x, &pool).is_none());
    }
}
