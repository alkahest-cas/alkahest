//! Integrand pre-normalisation and closed forms with symbolic parameters.
//!
//! # Why this module exists
//!
//! Every route in [`crate::integrate::integrate`] reads the integrand in the
//! shape the caller wrote it. That made a long list of textbook integrals fail
//! for reasons that have nothing to do with the mathematics:
//!
//! * `(σz)²·e^{−z²/2}` — the Risch tower sees the coefficient `(σz)²` and does
//!   not recognise it as `σ²·z²`;
//! * `(1 + x²/L²)²` — no rule expands a power of a sum;
//! * `e^{−aτ}·e^{−b(t−τ)}` — two exponentials whose product is one exponential
//!   of a linear argument;
//! * `x·x^{n−1}` and `τ·(aτ)^{−1}` — powers of the same base never combined;
//! * `x·e^{−kx}` — the tower's Risch DE works over `ℚ`, so a symbolic rate is
//!   "transcendental" to it (while `x·e^{kx}` happened to reach another route);
//! * `1/(k² + m²)` — the rational routes need numeric coefficients.
//!
//! This module runs **only on the decline path** of the dispatcher, so every
//! integral the existing routes already solve keeps its exact answer and its
//! cost. It does two things:
//!
//! 1. **Normalises** the integrand into a sum of flat products: integer powers
//!    of sums are expanded (bounded), products of exponentials are merged
//!    into one exponential whose argument is collected by powers of the
//!    variable, and powers of a common base are combined. Every rewrite is an
//!    identity of functions (`b^p·b^q = b^{p+q}` and `(uv)^n = u^n v^n` for
//!    integer `n` hold on the principal branch).
//! 2. **Integrates termwise**, with closed forms that accept symbolic
//!    parameters: `x^n·e^{αx+β}·{1, sin, cos}(ωx+φ)` by the tabular
//!    (repeated by-parts) formula, `x^s` for a symbolic `s`, and
//!    `(px+q)/(ax²+bx+c)` by completing the square — arctangent when the
//!    discriminant `4ac − b²` is provably positive, logarithms when provably
//!    negative, and the arctangent with the stated side condition
//!    `4ac − b² > 0` when its sign is unknown. A term no rule matches is handed
//!    back to the full integrator.
//!
//! Every division by a parameter is recorded as a side condition
//! ([`SideCondition::NonZero`]) unless the parameter's sign is established,
//! and the assembled antiderivative is checked against the **original**
//! integrand — exactly, or numerically at parameter values that satisfy the
//! side conditions — before it is returned. A candidate that fails the check
//! is a decline, never an answer.
//!
//! [`try_gamma_family`] is the definite-integral half: `∫₀^∞ x^{s−1}e^{−ax} dx
//! = Γ(s)/a^s` for `Re s > 0`, `a > 0`, applied termwise to a finite sum. Both
//! hypotheses must be *established* (numerically or from the assumptions);
//! otherwise it declines and the caller's ordinary route reports the missing
//! sign.

use super::engine::{
    classify_sample, collect_gate_parameters, is_free_of, verify_antiderivative_exact,
    IntegrationError, SampleVerdict, GATE_X_SAMPLES,
};
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use crate::simplify::{assumed_sign, Sign};
use rug::ops::Pow;
use std::cell::Cell;
use std::collections::HashMap;

/// Most summands an expansion may produce before the pass gives up.
const MAX_TERMS: usize = 256;
/// Largest integer power of a sum that is expanded.
const MAX_EXPAND_POWER: i64 = 12;
/// Largest polynomial degree the tabular formula is applied to.
const MAX_TABULAR_DEGREE: i64 = 24;
/// Most free parameters the verification sweep will enumerate.
const MAX_VERIFY_PARAMS: usize = 4;

thread_local! {
    /// Set while this pass is running, so the full integrator it calls on a
    /// single term does not re-enter it (the term is already normalised).
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

struct ActiveGuard;

impl ActiveGuard {
    fn enter() -> Option<Self> {
        ACTIVE.with(|a| {
            if a.get() {
                None
            } else {
                a.set(true);
                Some(ActiveGuard)
            }
        })
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE.with(|a| a.set(false));
    }
}

// ---------------------------------------------------------------------------
// Small constructors and numeric helpers
// ---------------------------------------------------------------------------

fn mk_mul(mut f: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    match f.len() {
        0 => pool.integer(1_i32),
        1 => f.pop().unwrap(),
        _ => pool.mul(f),
    }
}

fn mk_add(mut t: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    match t.len() {
        0 => pool.integer(0_i32),
        1 => t.pop().unwrap(),
        _ => pool.add(t),
    }
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), e])
}

fn inv(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(e, pool.integer(-1_i32))
}

fn rat_expr(r: &rug::Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

/// The exact rational value of a numeric leaf (`Integer`/`Rational`).
fn as_rational(e: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

fn as_small_int(e: ExprId, pool: &ExprPool) -> Option<i64> {
    match pool.get(e) {
        ExprData::Integer(n) => n.0.to_i64(),
        _ => None,
    }
}

fn is_int(e: ExprId, pool: &ExprPool, v: i64) -> bool {
    as_small_int(e, pool) == Some(v)
}

/// The sign of a `var`-free quantity: numeric value first, then the ambient
/// assumptions and declared symbol domains.
fn sign_of(e: ExprId, pool: &ExprPool) -> Option<Sign> {
    let s = simplify(e, pool).value;
    if let Some(r) = as_rational(s, pool) {
        return Some(match r.cmp0() {
            std::cmp::Ordering::Less => Sign::Negative,
            std::cmp::Ordering::Equal => Sign::Zero,
            std::cmp::Ordering::Greater => Sign::Positive,
        });
    }
    if let Some(sg) = assumed_sign(s, pool) {
        return Some(sg);
    }
    // A closed constant (`π`, `√2`, `e`) has a sign its value shows.
    let mut params = Vec::new();
    collect_gate_parameters(s, pool.integer(0_i32), pool, &mut params);
    if params.is_empty() {
        if let Some(v) = crate::jit::eval_interp(s, &HashMap::new(), pool) {
            if v.is_finite() && v.abs() > 1e-12 {
                return Some(if v > 0.0 {
                    Sign::Positive
                } else {
                    Sign::Negative
                });
            }
        }
    }
    None
}

/// The side condition `e ≠ 0` a division by `e` needs: `Ok(None)` when it is
/// established, `Ok(Some(cond))` when it must be stated, `Err(())` when `e`
/// is identically zero (the formula does not apply at all).
fn nonzero_condition(e: ExprId, pool: &ExprPool) -> Result<Option<SideCondition>, ()> {
    let s = simplify(e, pool).value;
    match sign_of(s, pool) {
        Some(Sign::Zero) => Err(()),
        Some(_) => Ok(None),
        None => Ok(Some(SideCondition::NonZero(s))),
    }
}

/// An exact square root of `e` built from its structure: `g` for `g²`,
/// the rational root of a perfect-square rational, and products of those.
fn exact_sqrt(e: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if let Some(r) = as_rational(e, pool) {
        if r < 0 {
            return None;
        }
        let (n, d) = r.into_numer_denom();
        let (sn, rn) = n.sqrt_rem(rug::Integer::new());
        let (sd, rd) = d.sqrt_rem(rug::Integer::new());
        if rn == 0 && rd == 0 {
            return Some(rat_expr(&rug::Rational::from((sn, sd)), pool));
        }
        return None;
    }
    match pool.get(e) {
        ExprData::Pow { base, exp } => {
            let n = as_small_int(exp, pool)?;
            if n % 2 == 0 && n != 0 {
                let h = n / 2;
                if h == 1 {
                    Some(base)
                } else {
                    Some(pool.pow(base, pool.integer(h)))
                }
            } else {
                None
            }
        }
        ExprData::Mul(args) => {
            let mut out = Vec::with_capacity(args.len());
            for a in args {
                out.push(exact_sqrt(a, pool)?);
            }
            Some(mk_mul(out, pool))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Normalisation: expansion into a sum of flat products
// ---------------------------------------------------------------------------

type Term = Vec<ExprId>;

/// Multiply two expanded sums term by term, refusing past [`MAX_TERMS`].
fn product(a: &[Term], b: &[Term]) -> Option<Vec<Term>> {
    if a.len().saturating_mul(b.len()) > MAX_TERMS {
        return None;
    }
    let mut out = Vec::with_capacity(a.len() * b.len());
    for ta in a {
        for tb in b {
            let mut t = ta.clone();
            t.extend_from_slice(tb);
            out.push(t);
        }
    }
    Some(out)
}

/// Expand `e` into summands (each a list of factors). `var`-free subtrees are
/// atoms: parameters are never expanded, only the dependence on `var` is.
fn expand_terms(e: ExprId, var: ExprId, pool: &ExprPool, depth: u32) -> Option<Vec<Term>> {
    if depth > 64 {
        return None;
    }
    if is_free_of(e, var, pool) {
        return Some(vec![vec![e]]);
    }
    match pool.get(e) {
        ExprData::Add(args) => {
            let mut out = Vec::new();
            for a in args {
                out.extend(expand_terms(a, var, pool, depth + 1)?);
                if out.len() > MAX_TERMS {
                    return None;
                }
            }
            Some(out)
        }
        ExprData::Mul(args) => {
            let mut acc: Vec<Term> = vec![Vec::new()];
            for a in args {
                let ta = expand_terms(a, var, pool, depth + 1)?;
                acc = product(&acc, &ta)?;
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            if let Some(n) = as_small_int(exp, pool) {
                // (u·v)^n = u^n·v^n for integer n.
                if let ExprData::Mul(fs) = pool.get(base) {
                    let mut acc: Vec<Term> = vec![Vec::new()];
                    for f in fs {
                        let tf = expand_terms(pool.pow(f, exp), var, pool, depth + 1)?;
                        acc = product(&acc, &tf)?;
                    }
                    return Some(acc);
                }
                // (b^p)^n = b^{pn} for integer n.
                if let ExprData::Pow {
                    base: inner,
                    exp: p,
                } = pool.get(base)
                {
                    let pn = match as_rational(p, pool) {
                        Some(r) => rat_expr(&(r * rug::Rational::from(n)), pool),
                        None => simplify(pool.mul(vec![p, exp]), pool).value,
                    };
                    return expand_terms(pool.pow(inner, pn), var, pool, depth + 1);
                }
                if let ExprData::Func { name, args } = pool.get(base) {
                    if name == "exp" && args.len() == 1 {
                        // exp(u)^n = exp(n·u).
                        let arg = pool.mul(vec![exp, args[0]]);
                        return expand_terms(pool.func("exp", vec![arg]), var, pool, depth + 1);
                    }
                }
                let bt = expand_terms(base, var, pool, depth + 1)?;
                if bt.len() > 1 && (2..=MAX_EXPAND_POWER).contains(&n) {
                    let mut acc = bt.clone();
                    for _ in 1..n {
                        acc = product(&acc, &bt)?;
                    }
                    return Some(acc);
                }
                let nb = rebuild(bt, var, pool);
                return Some(vec![vec![pool.pow(nb, exp)]]);
            }
            let nb = normalise(base, var, pool);
            let ne = normalise(exp, var, pool);
            Some(vec![vec![pool.pow(nb, ne)]])
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let a = normalise(args[0], var, pool);
            let a = if name == "exp" {
                collect_in_var(a, var, pool)
            } else {
                a
            };
            Some(vec![vec![pool.func(name, vec![a])]])
        }
        _ => Some(vec![vec![e]]),
    }
}

/// Reassemble expanded summands into one expression.
fn rebuild(terms: Vec<Term>, var: ExprId, pool: &ExprPool) -> ExprId {
    let mut out = Vec::with_capacity(terms.len());
    for t in terms {
        let p = normalise_product(t, var, pool);
        if !is_int(p, pool, 0) {
            out.push(p);
        }
    }
    mk_add(out, pool)
}

/// The normalised form of `e`, or `e` itself when expansion is refused.
pub(crate) fn normalise(e: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    if is_free_of(e, var, pool) {
        return e;
    }
    match expand_terms(e, var, pool, 0) {
        Some(t) => rebuild(t, var, pool),
        None => e,
    }
}

/// Flatten a product, fold its numeric factors, merge its exponentials into
/// one, and combine powers of a common base.
fn normalise_product(factors: Term, var: ExprId, pool: &ExprPool) -> ExprId {
    let mut flat = Vec::with_capacity(factors.len());
    let mut stack: Vec<ExprId> = factors.into_iter().rev().collect();
    while let Some(f) = stack.pop() {
        match pool.get(f) {
            ExprData::Mul(args) => stack.extend(args.into_iter().rev()),
            _ => flat.push(f),
        }
    }

    let mut coeff = rug::Rational::from(1);
    let mut exp_args: Vec<ExprId> = Vec::new();
    // (base, exponents) in first-seen order.
    let mut powers: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
    let one = pool.integer(1_i32);
    let half = pool.rational(1, 2);
    let push_power = |b: ExprId, e: ExprId, powers: &mut Vec<(ExprId, Vec<ExprId>)>| {
        if let Some(slot) = powers.iter_mut().find(|(pb, _)| *pb == b) {
            slot.1.push(e);
        } else {
            powers.push((b, vec![e]));
        }
    };
    for f in flat {
        if let Some(r) = as_rational(f, pool) {
            coeff *= r;
            continue;
        }
        match pool.get(f) {
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                exp_args.push(args[0]);
            }
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                push_power(args[0], half, &mut powers);
            }
            ExprData::Pow { base, exp } => {
                // A numeric power of a nonzero rational folds into the
                // coefficient (`2⁻¹`).
                if let (Some(b), Some(n)) = (as_rational(base, pool), as_small_int(exp, pool)) {
                    if b != 0 && n.unsigned_abs() <= 64 {
                        let p = rug::Rational::from(b.clone().pow(n.unsigned_abs() as u32));
                        coeff *= if n < 0 { p.recip() } else { p };
                        continue;
                    }
                }
                if let ExprData::Func { name, args } = pool.get(base) {
                    if name == "sqrt" && args.len() == 1 {
                        if let Some(r) = as_rational(exp, pool) {
                            let e = rat_expr(&(r / 2), pool);
                            push_power(args[0], e, &mut powers);
                            continue;
                        }
                    }
                }
                // (b^p)^n = b^{pn} for integer n.
                if let (
                    ExprData::Pow {
                        base: inner,
                        exp: p,
                    },
                    Some(n),
                ) = (pool.get(base), as_small_int(exp, pool))
                {
                    let pn = match as_rational(p, pool) {
                        Some(r) => rat_expr(&(r * rug::Rational::from(n)), pool),
                        None => simplify(pool.mul(vec![p, exp]), pool).value,
                    };
                    push_power(inner, pn, &mut powers);
                    continue;
                }
                push_power(base, exp, &mut powers);
            }
            _ => push_power(f, one, &mut powers),
        }
    }
    if coeff == 0 {
        return pool.integer(0_i32);
    }

    let mut out = Vec::new();
    if coeff != 1 {
        out.push(rat_expr(&coeff, pool));
    }
    for (b, es) in powers {
        let total = sum_exponents(&es, pool);
        if is_int(total, pool, 0) {
            continue;
        }
        if is_int(total, pool, 1) {
            out.push(b);
        } else if total == half && b != var {
            out.push(pool.func("sqrt", vec![b]));
        } else {
            out.push(pool.pow(b, total));
        }
    }
    if !exp_args.is_empty() {
        let arg = if exp_args.len() == 1 {
            exp_args[0]
        } else {
            let summed = pool.add(exp_args);
            collect_in_var(normalise(summed, var, pool), var, pool)
        };
        if !is_int(arg, pool, 0) {
            out.push(pool.func("exp", vec![arg]));
        }
    }
    mk_mul(out, pool)
}

fn sum_exponents(es: &[ExprId], pool: &ExprPool) -> ExprId {
    if es.len() == 1 {
        return es[0];
    }
    let mut acc = rug::Rational::new();
    let mut symbolic = Vec::new();
    for &e in es {
        match as_rational(e, pool) {
            Some(r) => acc += r,
            None => symbolic.push(e),
        }
    }
    if symbolic.is_empty() {
        return rat_expr(&acc, pool);
    }
    if acc != 0 {
        symbolic.push(rat_expr(&acc, pool));
    }
    simplify(mk_add(symbolic, pool), pool).value
}

/// The degree of `var` in a flat product and its `var`-free cofactor, or
/// `None` when the product is not a monomial in `var`.
fn monomial(t: ExprId, var: ExprId, pool: &ExprPool) -> Option<(i64, ExprId)> {
    let factors = match pool.get(t) {
        ExprData::Mul(a) => a,
        _ => vec![t],
    };
    let mut deg = 0_i64;
    let mut rest = Vec::new();
    for f in factors {
        if f == var {
            deg += 1;
            continue;
        }
        if let ExprData::Pow { base, exp } = pool.get(f) {
            if base == var {
                let n = as_small_int(exp, pool)?;
                if n < 0 {
                    return None;
                }
                deg += n;
                continue;
            }
        }
        if !is_free_of(f, var, pool) {
            return None;
        }
        rest.push(f);
    }
    Some((deg, mk_mul(rest, pool)))
}

/// The coefficients `[c₀, c₁, …]` of a polynomial in `var` with `var`-free
/// coefficients (already normalised), or `None`.
fn poly_coeffs(e: ExprId, var: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
    let e = normalise(e, var, pool);
    let terms = match pool.get(e) {
        ExprData::Add(a) => a,
        _ => vec![e],
    };
    let mut by_deg: Vec<Vec<ExprId>> = Vec::new();
    for t in terms {
        let (d, c) = monomial(t, var, pool)?;
        if d > 64 {
            return None;
        }
        let d = d as usize;
        if by_deg.len() <= d {
            by_deg.resize(d + 1, Vec::new());
        }
        by_deg[d].push(c);
    }
    let mut out: Vec<ExprId> = by_deg
        .into_iter()
        .map(|cs| simplify(mk_add(cs, pool), pool).value)
        .collect();
    while out.len() > 1 && is_int(*out.last().unwrap(), pool, 0) {
        out.pop();
    }
    Some(out)
}

/// Collect a sum by powers of `var` (`−aτ − bt + bτ` → `(b − a)·τ − bt`), so
/// a linear exponent is visible as one.
fn collect_in_var(e: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    let terms = match pool.get(e) {
        ExprData::Add(a) => a,
        _ => return e,
    };
    let mut by_deg: Vec<Vec<ExprId>> = Vec::new();
    let mut other = Vec::new();
    for t in terms {
        match monomial(t, var, pool) {
            Some((d, c)) if d <= 64 => {
                let d = d as usize;
                if by_deg.len() <= d {
                    by_deg.resize(d + 1, Vec::new());
                }
                by_deg[d].push(c);
            }
            _ => other.push(t),
        }
    }
    let mut out = other;
    for (d, cs) in by_deg.into_iter().enumerate() {
        if cs.is_empty() {
            continue;
        }
        let c = simplify(mk_add(cs, pool), pool).value;
        if is_int(c, pool, 0) {
            continue;
        }
        let mono = match d {
            0 => None,
            1 => Some(var),
            _ => Some(pool.pow(var, pool.integer(d as i64))),
        };
        out.push(match mono {
            None => c,
            Some(m) if is_int(c, pool, 1) => m,
            Some(m) => pool.mul(vec![c, m]),
        });
    }
    mk_add(out, pool)
}

/// `(α, β)` with `e = α·var + β` and `α` not identically zero.
fn linear_coeffs(e: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let c = poly_coeffs(e, var, pool)?;
    if c.len() != 2 || is_int(c[1], pool, 0) {
        return None;
    }
    Some((c[1], c[0]))
}

// ---------------------------------------------------------------------------
// Term shapes and their closed forms
// ---------------------------------------------------------------------------

/// A flat product read as `coeff · var^p · exp(αx+β) · trig(ωx+φ) / Q(x)`.
struct Shape {
    coeff: Vec<ExprId>,
    power: Option<ExprId>,
    exp: Option<(ExprId, ExprId, ExprId)>, // (generator, α, β)
    trig: Option<(bool, ExprId, ExprId, ExprId)>, // (is_sin, generator, ω, φ)
    recip: Option<ExprId>,                 // Q, with the factor Q⁻¹
    other: bool,
}

fn read_shape(t: ExprId, var: ExprId, pool: &ExprPool) -> Shape {
    let factors = match pool.get(t) {
        ExprData::Mul(a) => a,
        _ => vec![t],
    };
    let mut s = Shape {
        coeff: Vec::new(),
        power: None,
        exp: None,
        trig: None,
        recip: None,
        other: false,
    };
    for f in factors {
        if is_free_of(f, var, pool) {
            s.coeff.push(f);
            continue;
        }
        if f == var && s.power.is_none() {
            s.power = Some(pool.integer(1_i32));
            continue;
        }
        match pool.get(f) {
            ExprData::Pow { base, exp } if base == var && s.power.is_none() => {
                if is_free_of(exp, var, pool) {
                    s.power = Some(exp);
                    continue;
                }
            }
            ExprData::Pow { base, exp } if is_int(exp, pool, -1) && s.recip.is_none() => {
                s.recip = Some(base);
                continue;
            }
            ExprData::Func { name, args } if args.len() == 1 => {
                let lin = linear_coeffs(args[0], var, pool);
                match (name.as_str(), lin) {
                    ("exp", Some((a, b))) if s.exp.is_none() => {
                        s.exp = Some((f, a, b));
                        continue;
                    }
                    ("sin", Some((w, p))) if s.trig.is_none() => {
                        s.trig = Some((true, f, w, p));
                        continue;
                    }
                    ("cos", Some((w, p))) if s.trig.is_none() => {
                        s.trig = Some((false, f, w, p));
                        continue;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        s.other = true;
    }
    s
}

fn binomial(n: i64, k: i64) -> rug::Integer {
    rug::Integer::from(n).binomial(k as u32)
}

fn falling(n: i64, j: i64) -> rug::Integer {
    // n·(n−1)·…·(n−j+1)
    let mut r = rug::Integer::from(1);
    for i in 0..j {
        r *= n - i;
    }
    r
}

/// `∫ x^n · e^{αx+β} · {1, cos, sin}(ωx+φ) dx` by the tabular formula.
///
/// With `c = α + iω`, `∫ xⁿ e^{cx} dx = e^{cx} Σⱼ (−1)ʲ n!/(n−j)! x^{n−j}/c^{j+1}`;
/// `1/c^{m} = c̄^m/R^m` with `R = α² + ω²`, and the real and imaginary parts
/// of `e^{iB}·P` give the cosine and sine cases.
fn tabular(
    n: i64,
    exp: Option<(ExprId, ExprId, ExprId)>,
    trig: Option<(bool, ExprId, ExprId, ExprId)>,
    var: ExprId,
    pool: &ExprPool,
    conds: &mut Vec<SideCondition>,
) -> Option<ExprId> {
    let zero = pool.integer(0_i32);
    let alpha = exp.map(|e| e.1).unwrap_or(zero);
    let x_pow = |k: i64| match k {
        0 => pool.integer(1_i32),
        1 => var,
        _ => pool.pow(var, pool.integer(k)),
    };
    match trig {
        None => {
            let (gen, a, _) = exp?;
            conds.extend(nonzero_condition(a, pool).ok()?);
            let mut terms = Vec::new();
            for j in 0..=n {
                let c = falling(n, j);
                let c = if j % 2 == 1 { -c } else { c };
                terms.push(mk_mul(
                    vec![
                        pool.integer(c),
                        x_pow(n - j),
                        pool.pow(a, pool.integer(-(j + 1))),
                    ],
                    pool,
                ));
            }
            Some(pool.mul(vec![gen, mk_add(terms, pool)]))
        }
        Some((is_sin, tgen, w, _phi)) => {
            let has_alpha = exp.is_some();
            // R = α² + ω²; nonzero unless both vanish.
            let r = if has_alpha {
                pool.add(vec![
                    pool.pow(alpha, pool.integer(2_i32)),
                    pool.pow(w, pool.integer(2_i32)),
                ])
            } else {
                pool.pow(w, pool.integer(2_i32))
            };
            let alpha_known_nonzero =
                has_alpha && matches!(nonzero_condition(alpha, pool), Ok(None));
            if !alpha_known_nonzero {
                match nonzero_condition(w, pool) {
                    Ok(None) => {}
                    Ok(Some(_)) => conds.push(SideCondition::NonZero(simplify(r, pool).value)),
                    Err(()) => return None,
                }
            }
            let mut pr = Vec::new();
            let mut pi = Vec::new();
            for j in 0..=n {
                let m = j + 1;
                let c = falling(n, j);
                let c = if j % 2 == 1 { -c } else { c };
                let scale = vec![pool.integer(c), x_pow(n - j), pool.pow(r, pool.integer(-m))];
                // c̄^m = Σ_r C(m,r) α^{m−r} (−iω)^r
                for rr in 0..=m {
                    if !has_alpha && rr != m {
                        continue;
                    }
                    let b = binomial(m, rr);
                    let mut f = scale.clone();
                    if m - rr > 0 {
                        f.push(pool.pow(alpha, pool.integer(m - rr)));
                    }
                    if rr > 0 {
                        f.push(pool.pow(w, pool.integer(rr)));
                    }
                    // (−i)^r: 1, −i, −1, i
                    let (sign, real) = match rr % 4 {
                        0 => (1, true),
                        1 => (-1, false),
                        2 => (-1, true),
                        _ => (1, false),
                    };
                    f.push(pool.integer(b * sign));
                    if real {
                        pr.push(mk_mul(f, pool));
                    } else {
                        pi.push(mk_mul(f, pool));
                    }
                }
            }
            let pr = mk_add(pr, pool);
            let pi = mk_add(pi, pool);
            let (cos_b, sin_b) = if is_sin {
                (pool.func("cos", vec![trig_arg(tgen, pool)]), tgen)
            } else {
                (tgen, pool.func("sin", vec![trig_arg(tgen, pool)]))
            };
            // cos: cosB·Pr − sinB·Pi ; sin: sinB·Pr + cosB·Pi
            let body = if is_sin {
                pool.add(vec![pool.mul(vec![sin_b, pr]), pool.mul(vec![cos_b, pi])])
            } else {
                pool.add(vec![
                    pool.mul(vec![cos_b, pr]),
                    neg(pool.mul(vec![sin_b, pi]), pool),
                ])
            };
            Some(match exp {
                Some((gen, _, _)) => pool.mul(vec![gen, body]),
                None => body,
            })
        }
    }
}

fn trig_arg(gen: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(gen) {
        ExprData::Func { args, .. } => args[0],
        _ => gen,
    }
}

/// `∫ x^p / Q(x) dx` for `p ∈ {0, 1}` and `Q` of degree 1 or 2 with
/// `var`-free (possibly symbolic) coefficients.
fn rational_low_degree(
    p: i64,
    q: ExprId,
    var: ExprId,
    pool: &ExprPool,
    conds: &mut Vec<SideCondition>,
) -> Option<ExprId> {
    let c = poly_coeffs(q, var, pool)?;
    match c.len() {
        2 => {
            let (alpha, beta) = (c[1], c[0]);
            conds.extend(nonzero_condition(alpha, pool).ok()?);
            let lg = pool.func("log", vec![q]);
            match p {
                0 => Some(pool.mul(vec![inv(alpha, pool), lg])),
                1 => Some(pool.add(vec![
                    pool.mul(vec![var, inv(alpha, pool)]),
                    neg(
                        pool.mul(vec![beta, pool.pow(alpha, pool.integer(-2_i32)), lg]),
                        pool,
                    ),
                ])),
                _ => None,
            }
        }
        3 => {
            let (cc, b, a) = (c[0], c[1], c[2]);
            conds.extend(nonzero_condition(a, pool).ok()?);
            let base = reciprocal_quadratic(a, b, cc, q, var, pool, conds)?;
            match p {
                0 => Some(base),
                1 => {
                    // x/Q = (1/2a)(2ax+b)/Q − (b/2a)/Q
                    let two_a_inv = inv(pool.mul(vec![pool.integer(2_i32), a]), pool);
                    let lg = pool.mul(vec![two_a_inv, pool.func("log", vec![q])]);
                    if is_int(b, pool, 0) {
                        Some(lg)
                    } else {
                        Some(pool.add(vec![lg, neg(pool.mul(vec![b, two_a_inv, base]), pool)]))
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// `∫ dx / (ax² + bx + c)` by completing the square.
fn reciprocal_quadratic(
    a: ExprId,
    b: ExprId,
    c: ExprId,
    q: ExprId,
    var: ExprId,
    pool: &ExprPool,
    conds: &mut Vec<SideCondition>,
) -> Option<ExprId> {
    let two = pool.integer(2_i32);
    let four = pool.integer(4_i32);
    // b = 0 and a, c exact squares (`k² + m²`, `α²x² + γ²`): the arctangent is
    // real for every real nonzero parameter, so no sign condition is needed.
    if is_int(b, pool, 0) {
        if let (Some(ra), Some(rc)) = (exact_sqrt(a, pool), exact_sqrt(c, pool)) {
            let r = simplify(pool.mul(vec![ra, rc]), pool).value;
            conds.extend(nonzero_condition(r, pool).ok()?);
            // atan(a·x/r)/r, r² = a·c
            let arg = simplify(pool.mul(vec![a, var, inv(r, pool)]), pool).value;
            return Some(pool.mul(vec![inv(r, pool), pool.func("atan", vec![arg])]));
        }
        // b = 0 otherwise: atan(a·x/√(ac))/√(ac), real when ac > 0.
        let ac = simplify(pool.mul(vec![a, c]), pool).value;
        match sign_of(ac, pool) {
            Some(Sign::Zero) => return None,
            Some(Sign::Negative) => {}
            sign => {
                if sign.is_none() {
                    conds.push(SideCondition::Positive(ac));
                }
                let r = pool.func("sqrt", vec![ac]);
                let arg = pool.mul(vec![a, var, inv(r, pool)]);
                return Some(pool.mul(vec![inv(r, pool), pool.func("atan", vec![arg])]));
            }
        }
    }
    // D = 4ac − b²
    let d = simplify(
        pool.add(vec![
            pool.mul(vec![four, a, c]),
            neg(pool.pow(b, two), pool),
        ]),
        pool,
    )
    .value;
    let lin = pool.add(vec![pool.mul(vec![two, a, var]), b]); // 2ax + b
    let sqrt_of = |e: ExprId| exact_sqrt(e, pool).unwrap_or_else(|| pool.func("sqrt", vec![e]));
    match sign_of(d, pool) {
        Some(Sign::Zero) => {
            // a(x + b/2a)² : ∫ = −2/(2ax + b)
            let _ = q;
            Some(pool.mul(vec![pool.integer(-2_i32), inv(lin, pool)]))
        }
        Some(Sign::Negative) => {
            let nd = simplify(neg(d, pool), pool).value;
            let s = sqrt_of(nd);
            let l1 = pool.func("log", vec![pool.add(vec![lin, neg(s, pool)])]);
            let l2 = pool.func("log", vec![pool.add(vec![lin, s])]);
            Some(pool.mul(vec![inv(s, pool), pool.add(vec![l1, neg(l2, pool)])]))
        }
        sign => {
            if sign.is_none() {
                conds.push(SideCondition::Positive(d));
            }
            let s = sqrt_of(d);
            let arg = pool.mul(vec![lin, inv(s, pool)]);
            Some(pool.mul(vec![two, inv(s, pool), pool.func("atan", vec![arg])]))
        }
    }
}

/// Closed form for one normalised term, or `None` when no rule here applies.
fn integrate_shape(
    t: ExprId,
    var: ExprId,
    pool: &ExprPool,
    conds: &mut Vec<SideCondition>,
) -> Option<ExprId> {
    let s = read_shape(t, var, pool);
    if s.other {
        return None;
    }
    let coeff = mk_mul(s.coeff.clone(), pool);
    let with_coeff = |f: ExprId| {
        if is_int(coeff, pool, 1) {
            f
        } else {
            pool.mul(vec![coeff, f])
        }
    };
    let n_int = match s.power {
        None => Some(0),
        Some(p) => as_small_int(p, pool),
    };

    if let Some(q) = s.recip {
        if s.exp.is_some() || s.trig.is_some() {
            return None;
        }
        let n = n_int?;
        if !(0..=1).contains(&n) {
            return None;
        }
        return rational_low_degree(n, q, var, pool, conds).map(with_coeff);
    }

    if s.exp.is_none() && s.trig.is_none() {
        // ∫ x^p dx
        let p = s.power?;
        if is_int(p, pool, -1) {
            return Some(with_coeff(pool.func("log", vec![var])));
        }
        let p1 = simplify(pool.add(vec![p, pool.integer(1_i32)]), pool).value;
        conds.extend(nonzero_condition(p1, pool).ok()?);
        return Some(with_coeff(pool.mul(vec![pool.pow(var, p1), inv(p1, pool)])));
    }

    let n = n_int?;
    if !(0..=MAX_TABULAR_DEGREE).contains(&n) {
        return None;
    }
    tabular(n, s.exp, s.trig, var, pool, conds).map(with_coeff)
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

const PARAM_POSITIVE: [f64; 3] = [0.6131, 1.3719, 2.8123];
const PARAM_GENERAL: [f64; 4] = [-1.7231, 0.6131, 2.8123, -0.4517];

fn condition_holds(c: &SideCondition, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> bool {
    match c {
        SideCondition::NonZero(e) => {
            matches!(crate::jit::eval_interp(*e, env, pool), Some(v) if v.is_finite() && v.abs() > 1e-9)
        }
        SideCondition::Positive(e) => {
            matches!(crate::jit::eval_interp(*e, env, pool), Some(v) if v.is_finite() && v > 1e-9)
        }
        SideCondition::InDomain(..) => false,
    }
}

/// `F′ = f` exactly, or numerically at every sampled parameter point that
/// satisfies the side conditions (at least one such point, and a quorum of
/// agreeing `var` samples at each).
fn verify(
    f_cand: ExprId,
    integrand: ExprId,
    var: ExprId,
    conds: &[SideCondition],
    pool: &ExprPool,
) -> bool {
    if verify_antiderivative_exact(f_cand, integrand, var, pool) {
        return true;
    }
    let Ok(d_raw) = crate::diff::diff(f_cand, var, pool) else {
        return false;
    };
    let d = simplify(d_raw.value, pool).value;
    let mut params = Vec::new();
    collect_gate_parameters(f_cand, var, pool, &mut params);
    collect_gate_parameters(integrand, var, pool, &mut params);
    if params.len() > MAX_VERIFY_PARAMS {
        return false;
    }
    let grids: Vec<Vec<f64>> = params
        .iter()
        .map(|&p| match pool.get(p) {
            ExprData::Symbol { domain, .. }
                if matches!(
                    domain,
                    crate::kernel::Domain::Positive | crate::kernel::Domain::NonNegative
                ) =>
            {
                PARAM_POSITIVE.to_vec()
            }
            _ => PARAM_GENERAL.to_vec(),
        })
        .collect();
    let combos: usize = grids.iter().map(|g| g.len()).product();
    let mut checked_combos = 0;
    for idx in 0..combos {
        let mut env = HashMap::new();
        let mut rest = idx;
        for (slot, g) in grids.iter().enumerate() {
            env.insert(params[slot], g[rest % g.len()]);
            rest /= g.len();
        }
        if !conds.iter().all(|c| condition_holds(c, &env, pool)) {
            continue;
        }
        let mut agree = 0;
        for &xv in &GATE_X_SAMPLES {
            let mut e2 = env.clone();
            e2.insert(var, xv);
            match classify_sample(d, integrand, &e2, pool) {
                SampleVerdict::Agrees => agree += 1,
                SampleVerdict::NoInformation => {}
                SampleVerdict::Disagrees => return false,
            }
        }
        if agree < 2 {
            return false;
        }
        checked_combos += 1;
    }
    checked_combos > 0
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// The decline-path tier: normalise, integrate termwise, verify.
///
/// `Ok(None)` is a decline (the caller keeps its own diagnostic); `Err` is
/// only a budget trip.
pub(crate) fn try_prenormalised(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<Option<DerivedExpr<ExprId>>, IntegrationError> {
    let Some(_guard) = ActiveGuard::enter() else {
        return Ok(None);
    };
    if !crate::kernel::expr_props::mult_tree_is_commutative(pool, expr) {
        return Ok(None);
    }
    crate::budget::check()?;
    let Some(terms) = expand_terms(expr, var, pool, 0) else {
        return Ok(None);
    };
    let terms: Vec<ExprId> = terms
        .into_iter()
        .map(|t| normalise_product(t, var, pool))
        .filter(|&t| !is_int(t, pool, 0))
        .collect();

    let mut conds: Vec<SideCondition> = Vec::new();
    let mut pieces = Vec::with_capacity(terms.len());
    let mut matched_any = false;
    let mut pending: Vec<ExprId> = Vec::new();
    for &t in &terms {
        crate::budget::check()?;
        if is_free_of(t, var, pool) {
            pieces.push(pool.mul(vec![t, var]));
            continue;
        }
        match integrate_shape(t, var, pool, &mut conds) {
            Some(f) => {
                matched_any = true;
                pieces.push(f);
            }
            None => pending.push(t),
        }
    }
    // Nothing changed shape and nothing matched: the full integrator already
    // saw exactly this, so asking it again would only repeat its decline.
    if !matched_any && terms.len() == 1 && terms[0] == expr {
        return Ok(None);
    }
    for t in pending {
        crate::budget::check()?;
        match super::engine::integrate(t, var, pool) {
            Ok(r) => pieces.push(r.value),
            Err(e) if e.is_budget() => return Err(e),
            Err(_) => return Ok(None),
        }
    }
    let raw = mk_add(pieces, pool);
    let f = simplify(raw, pool).value;
    // Deduplicate the conditions (several terms can need the same one).
    let mut uniq: Vec<SideCondition> = Vec::new();
    for c in conds {
        if !uniq.contains(&c) {
            uniq.push(c);
        }
    }
    crate::budget::check()?;
    if !verify(f, expr, var, &uniq, pool) {
        return Ok(None);
    }
    // The outermost `integrate` refuses an answer that is right only where the
    // integrand is real. Ask the same question here, so such a candidate is
    // this tier's decline — and the tiers after it (by-parts, which closes
    // `∫dx/(√x·√(1+x))` as `2·asinh(√x)`) still get their turn.
    if super::branch_check::complex_branch_mismatch(f, expr, var, pool).is_some() {
        return Ok(None);
    }
    let mut log = DerivationLog::new();
    log.push(RewriteStep::with_conditions(
        "int_prenormalise_termwise",
        expr,
        f,
        uniq,
    ));
    Ok(Some(DerivedExpr::with_log(f, log)))
}

/// `Γ(s)` for an exact `s > 0`: `(s−1)!` at integers, `(2m)!/(4^m m!)·√π` at
/// half-integers, the symbolic `gamma(s)` otherwise.
fn gamma_value(s: ExprId, pool: &ExprPool) -> ExprId {
    if let Some(r) = as_rational(s, pool) {
        if *r.denom() == 1 {
            if let Some(n) = r.numer().to_u32() {
                if (1..=170).contains(&n) {
                    return pool.integer(rug::Integer::from(rug::Integer::factorial(n - 1)));
                }
            }
        } else if *r.denom() == 2 {
            // s = m + 1/2
            let m = (r.clone() - rug::Rational::from((1, 2))).numer().to_u32();
            if let Some(m) = m.filter(|&m| m <= 170) {
                let num = rug::Integer::from(rug::Integer::factorial(2 * m));
                let den =
                    rug::Integer::from(4).pow(m) * rug::Integer::from(rug::Integer::factorial(m));
                let c = rug::Rational::from((num, den));
                let sqrt_pi = pool.func("sqrt", vec![crate::eval::symbols::pi_symbol(pool)]);
                return mk_mul(vec![rat_expr(&c, pool), sqrt_pi], pool);
            }
        }
    }
    pool.func("gamma", vec![s])
}

/// `rate^{−s}`, exact for a rational rate and an integer or half-integer
/// `s` (`(1/2)^{−1/2} = √2`), symbolic otherwise.
fn rate_power(rate: ExprId, s: ExprId, pool: &ExprPool) -> ExprId {
    if let (Some(r), Some(sv)) = (as_rational(rate, pool), as_rational(s, pool)) {
        if r > 0 && (*sv.denom() == 1 || *sv.denom() == 2) {
            let fl = sv.clone().floor();
            if let Some(m) = fl.numer().to_i32().filter(|m| m.unsigned_abs() <= 64) {
                let ri = r.clone().recip();
                let mut c = rug::Rational::from(ri.clone().pow(m.unsigned_abs()));
                if m < 0 {
                    c = c.recip();
                }
                let mut out = vec![rat_expr(&c, pool)];
                if *sv.denom() == 2 {
                    out.push(pool.func("sqrt", vec![rat_expr(&ri, pool)]));
                }
                return simplify(mk_mul(out, pool), pool).value;
            }
        }
    }
    pool.pow(rate, neg(s, pool))
}

/// `∫₀^∞ Σ cᵢ·x^{pᵢ}·e^{αᵢx+βᵢ} dx = Σ cᵢ·e^{βᵢ}·Γ(pᵢ+1)/(−αᵢ)^{pᵢ+1}`.
///
/// Every term must have `pᵢ + 1 > 0` and `αᵢ < 0` *established* — from its
/// numeric value or the assumptions — or the whole integral declines
/// (`None`), and the caller's ordinary route reports what it is missing.
pub(crate) fn try_gamma_family(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<DerivedExpr<ExprId>> {
    if !crate::kernel::expr_props::mult_tree_is_commutative(pool, expr) {
        return None;
    }
    let terms = expand_terms(expr, var, pool, 0)?;
    let mut parts = Vec::new();
    for t in terms {
        let t = normalise_product(t, var, pool);
        if is_int(t, pool, 0) {
            continue;
        }
        let s = read_shape(t, var, pool);
        if s.other || s.trig.is_some() || s.recip.is_some() {
            return None;
        }
        let (gen, alpha, beta) = s.exp?;
        let _ = gen;
        let p = s.power.unwrap_or(pool.integer(0_i32));
        let sp = simplify(pool.add(vec![p, pool.integer(1_i32)]), pool).value;
        if sign_of(sp, pool) != Some(Sign::Positive) {
            return None;
        }
        let rate = simplify(neg(alpha, pool), pool).value;
        if sign_of(rate, pool) != Some(Sign::Positive) {
            return None;
        }
        let mut f = s.coeff;
        if !is_int(beta, pool, 0) {
            f.push(pool.func("exp", vec![beta]));
        }
        f.push(gamma_value(sp, pool));
        f.push(rate_power(rate, sp, pool));
        parts.push(mk_mul(f, pool));
    }
    if parts.is_empty() {
        return None;
    }
    let v = simplify(mk_add(parts, pool), pool);
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "gamma_function_integral",
        expr,
        v.value,
    ));
    Some(DerivedExpr::with_log(v.value, log.merge(v.log)))
}

/// Limit of `f` at an infinite `bound`, taken summand by summand after
/// normalisation: the limit of a sum is the sum of the limits **when each one
/// exists and is finite**, which is the only case accepted.
pub(crate) fn termwise_limit(
    f: ExprId,
    var: ExprId,
    bound: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let terms = expand_terms(f, var, pool, 0)?;
    if terms.len() < 2 {
        return None;
    }
    let mut out = Vec::with_capacity(terms.len());
    for t in terms {
        let t = normalise_product(t, var, pool);
        if is_free_of(t, var, pool) {
            out.push(t);
            continue;
        }
        let l = crate::calculus::limit(
            t,
            var,
            bound,
            crate::calculus::LimitDirection::Bidirectional,
            pool,
        )
        .ok()?;
        if !is_free_of(l, var, pool) {
            return None;
        }
        match crate::jit::eval_interp(l, &HashMap::new(), pool) {
            Some(v) if !v.is_finite() => return None,
            _ => {}
        }
        if l == pool.pos_infinity() || contains_infinity(l, pool) {
            return None;
        }
        out.push(l);
    }
    Some(simplify(mk_add(out, pool), pool).value)
}

fn contains_infinity(e: ExprId, pool: &ExprPool) -> bool {
    if e == pool.pos_infinity() {
        return true;
    }
    match pool.get(e) {
        ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
            a.iter().any(|&x| contains_infinity(x, pool))
        }
        ExprData::Pow { base, exp } => {
            contains_infinity(base, pool) || contains_infinity(exp, pool)
        }
        _ => false,
    }
}

/// The exponent of a `0^e` left in `e` with a non-numeric exponent, if any.
pub(crate) fn unresolved_zero_power(e: ExprId, pool: &ExprPool) -> Option<ExprId> {
    match pool.get(e) {
        ExprData::Pow { base, exp } => {
            if is_int(base, pool, 0) && as_rational(exp, pool).is_none() {
                return Some(exp);
            }
            unresolved_zero_power(base, pool).or_else(|| unresolved_zero_power(exp, pool))
        }
        ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
            a.iter().find_map(|&t| unresolved_zero_power(t, pool))
        }
        _ => None,
    }
}

/// Replace `0^e` by `0` where `e > 0` is established, and `1^e` by `1`. Used on endpoint values
/// (`F(0)` of `x^{n+1}/(n+1)` with `n > −1` assumed).
pub(crate) fn resolve_zero_powers(e: ExprId, pool: &ExprPool) -> ExprId {
    fn go(e: ExprId, pool: &ExprPool, changed: &mut bool) -> ExprId {
        match pool.get(e) {
            ExprData::Pow { base, exp } => {
                let b = go(base, pool, changed);
                let x = go(exp, pool, changed);
                if is_int(b, pool, 0)
                    && as_rational(x, pool).is_none()
                    && sign_of(x, pool) == Some(Sign::Positive)
                {
                    *changed = true;
                    return pool.integer(0_i32);
                }
                // 1^e = exp(e·log 1) = 1 for every e.
                if is_int(b, pool, 1) && as_rational(x, pool).is_none() {
                    *changed = true;
                    return b;
                }
                if b != base || x != exp {
                    pool.pow(b, x)
                } else {
                    e
                }
            }
            ExprData::Add(a) => {
                let n: Vec<_> = a.iter().map(|&t| go(t, pool, changed)).collect();
                if n != a {
                    pool.add(n)
                } else {
                    e
                }
            }
            ExprData::Mul(a) => {
                let n: Vec<_> = a.iter().map(|&t| go(t, pool, changed)).collect();
                if n != a {
                    pool.mul(n)
                } else {
                    e
                }
            }
            ExprData::Func { name, args } => {
                let n: Vec<_> = args.iter().map(|&t| go(t, pool, changed)).collect();
                if n != args {
                    pool.func(name, n)
                } else {
                    e
                }
            }
            _ => e,
        }
    }
    let mut changed = false;
    let r = go(e, pool, &mut changed);
    if changed {
        simplify(r, pool).value
    } else {
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::{integrate, integrate_definite};
    use crate::kernel::Domain;

    fn eval(e: ExprId, env: &[(ExprId, f64)], pool: &ExprPool) -> f64 {
        let m: HashMap<ExprId, f64> = env.iter().copied().collect();
        crate::jit::eval_interp(e, &m, pool).expect("evaluable")
    }

    /// `F(x₁) − F(x₀)` against a midpoint-rule quadrature of `f`.
    fn assert_antiderivative(
        f_anti: ExprId,
        f: ExprId,
        var: ExprId,
        params: &[(ExprId, f64)],
        pool: &ExprPool,
    ) {
        let (x0, x1) = (0.3, 1.7);
        let n = 4000;
        let h = (x1 - x0) / n as f64;
        let mut q = 0.0;
        for i in 0..n {
            let mut env = params.to_vec();
            env.push((var, x0 + (i as f64 + 0.5) * h));
            q += eval(f, &env, pool) * h;
        }
        let mut e0 = params.to_vec();
        e0.push((var, x0));
        let mut e1 = params.to_vec();
        e1.push((var, x1));
        let diff = eval(f_anti, &e1, pool) - eval(f_anti, &e0, pool);
        assert!(
            (diff - q).abs() < 1e-6 * (1.0 + q.abs()),
            "{}: F-difference {diff} vs quadrature {q}",
            pool.display(f_anti)
        );
    }

    #[test]
    fn poly_times_decaying_exp_with_symbolic_rate() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let k = pool.symbol("k", Domain::Real);
        let f = pool.mul(vec![
            x,
            pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), k, x])]),
        ]);
        let r = integrate(f, x, &pool).expect("∫ x·e^{−kx} dx");
        assert_antiderivative(r.value, f, x, &[(k, 1.3)], &pool);
        assert_antiderivative(r.value, f, x, &[(k, -0.7)], &pool);
    }

    #[test]
    fn reciprocal_sum_of_squares() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let m = pool.symbol("m", Domain::Real);
        let two = pool.integer(2_i32);
        let f = inv(pool.add(vec![pool.pow(k, two), pool.pow(m, two)]), &pool);
        let r = integrate(f, k, &pool).expect("∫ dk/(k²+m²)");
        assert!(format!("{}", pool.display(r.value)).contains("atan"));
        assert_antiderivative(r.value, f, k, &[(m, 0.8)], &pool);
        assert_antiderivative(r.value, f, k, &[(m, -1.9)], &pool);
    }

    #[test]
    fn reciprocal_quadratic_unknown_sign_states_condition() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let f = inv(
            pool.add(vec![pool.mul(vec![a, pool.pow(x, pool.integer(2_i32))]), b]),
            &pool,
        );
        let r = integrate(f, x, &pool).expect("∫ dx/(ax²+b)");
        let conds: Vec<_> = r
            .log
            .steps()
            .iter()
            .flat_map(|s| s.side_conditions.clone())
            .collect();
        assert!(
            conds
                .iter()
                .any(|c| matches!(c, SideCondition::Positive(_))),
            "the arctangent form needs 4ab > 0 stated: {conds:?}"
        );
        assert_antiderivative(r.value, f, x, &[(a, 1.1), (b, 2.3)], &pool);
    }

    #[test]
    fn power_of_sum_is_expanded() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let l = pool.symbol("L", Domain::Real);
        let two = pool.integer(2_i32);
        let inner = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.pow(x, two), pool.pow(l, pool.integer(-2_i32))]),
        ]);
        let f = pool.pow(inner, two);
        let r = integrate(f, x, &pool).expect("∫ (1 + x²/L²)² dx");
        assert_antiderivative(r.value, f, x, &[(l, 1.7)], &pool);
    }

    #[test]
    fn exponential_products_merge() {
        let pool = ExprPool::new();
        let tau = pool.symbol("tau", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let m1 = pool.integer(-1_i32);
        let f = pool.mul(vec![
            pool.func("exp", vec![pool.mul(vec![m1, a, tau])]),
            pool.func(
                "exp",
                vec![pool.mul(vec![m1, b, pool.add(vec![t, pool.mul(vec![m1, tau])])])],
            ),
        ]);
        let r = integrate(f, tau, &pool).expect("∫ e^{−aτ}e^{−b(t−τ)} dτ");
        assert_antiderivative(r.value, f, tau, &[(a, 0.6), (b, 1.9), (t, 0.4)], &pool);
    }

    #[test]
    fn tabular_exp_sin() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for n in 0..=3 {
            let f = pool.mul(vec![
                pool.pow(x, pool.integer(n)),
                pool.func("exp", vec![pool.mul(vec![pool.integer(2_i32), x])]),
                pool.func("sin", vec![x]),
            ]);
            let r = integrate(f, x, &pool).expect("∫ xⁿ e^{2x} sin x dx");
            assert_antiderivative(r.value, f, x, &[], &pool);
        }
    }

    #[test]
    fn gamma_family_chi_square() {
        let pool = ExprPool::new();
        let y = pool.symbol("y", Domain::Real);
        let f = pool.mul(vec![
            pool.func("exp", vec![pool.mul(vec![pool.rational(-1, 2), y])]),
            pool.pow(pool.func("sqrt", vec![y]), pool.integer(-1_i32)),
        ]);
        let r = integrate_definite(f, y, pool.integer(0_i32), pool.pos_infinity(), &pool)
            .expect("∫₀^∞ e^{−y/2}/√y dy");
        let v = eval(r.value, &[], &pool);
        assert!(
            (v - (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-12,
            "{v}"
        );
    }

    #[test]
    fn bateman_auc_termwise() {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Positive);
        let ke = pool.symbol("ke", Domain::Positive);
        let ka = pool.symbol("ka", Domain::Positive);
        let m1 = pool.integer(-1_i32);
        let decay = |k| pool.func("exp", vec![pool.mul(vec![m1, k, t])]);
        let f = pool.add(vec![decay(ke), pool.mul(vec![m1, decay(ka)])]);
        let r = integrate_definite(f, t, pool.integer(0_i32), pool.pos_infinity(), &pool)
            .expect("Bateman AUC");
        let v = eval(r.value, &[(ke, 0.3), (ka, 1.7)], &pool);
        assert!((v - (1.0 / 0.3 - 1.0 / 1.7)).abs() < 1e-12, "{v}");
        // The termwise limit on its own (the FTC route's fallback).
        let anti = integrate(f, t, &pool).expect("antiderivative");
        let l = termwise_limit(anti.value, t, pool.pos_infinity(), &pool).expect("limit");
        assert!(eval(l, &[(ke, 0.3), (ka, 1.7)], &pool).abs() < 1e-12);
    }

    #[test]
    fn unknown_rate_sign_refuses() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let f = pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), a, x])]);
        assert!(try_gamma_family(f, x, &pool).is_none());
        assert!(integrate_definite(f, x, pool.integer(0_i32), pool.pos_infinity(), &pool).is_err());
    }
}
