//! Closed forms Gosper's `Q(k)` machinery cannot reach (G6).
//!
//! Two families, both summed exactly and both reported with the side
//! conditions their closed forms need:
//!
//! 1. **Polynomial × geometric with a symbolic ratio,**
//!    `Σ_{k=lo}^{hi} P(k)·q^k` where `P` is a polynomial in `k` (any
//!    coefficients free of `k`) and `q` is any expression free of `k` —
//!    `r`, `exp(−λτ)`, `q^2`, a rational number. The antidifference is
//!    `Q(k)·q^k` with `deg Q = deg P`, solved for in closed form:
//!    `q·Q(k+1) − Q(k) = P(k)` is triangular in the coefficients of `Q`, with
//!    `q − 1` as the only divisor. A finite sum is `Q(hi+1)q^{hi+1} − Q(lo)q^{lo}`
//!    (side condition `q − 1 ≠ 0`); the sum to `∞` is `−Q(lo)q^{lo}` and needs
//!    `|q| < 1`, which is decided from the assumptions when it can be, refused
//!    when it provably fails and otherwise returned as a side condition.
//!
//! 2. **Convergent rational series to `∞`,** `Σ_{k=lo}^{∞} R(k)` with every
//!    pole of `R` linear in `k`. `R` is split into partial fractions
//!    `Σ c_{ρ,s}/(k−ρ)^s`; poles whose difference is an integer are one class
//!    and telescope against each other, leaving finitely many rational terms;
//!    what does not telescope is a Hurwitz zeta value, returned exactly as a
//!    rational multiple of `π^{2m}` for an even order at an integer or
//!    half-integer point, as a rational multiple of `zeta(s)` for an odd order
//!    there (`Σ 1/k³ = ζ(3)`), as `digamma`/`trigamma` at any other point, and
//!    as `hurwitz_zeta(s, x)` for a higher order at any other point.

use super::expr_ratio::is_free_of_k;
use super::special::zeta_even_coefficient;
use crate::deriv::log::SideCondition;
use crate::kernel::subs::subs;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use crate::simplify::{assumed_sign, Sign};
use rug::{Integer, Rational};
use std::collections::HashMap;

/// Largest polynomial degree in `k` handled (numerator of either family).
const MAX_DEGREE: usize = 24;
/// Largest number of explicit terms a telescoped remainder may have.
const MAX_TELESCOPE_TERMS: i64 = 10_000;

pub(super) fn simp(pool: &ExprPool, e: ExprId) -> ExprId {
    simplify(e, pool).value
}

pub(super) fn neg(pool: &ExprPool, e: ExprId) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), e])
}

pub(super) fn is_zero(pool: &ExprPool, e: ExprId) -> bool {
    match pool.get(e) {
        ExprData::Integer(n) => n.0 == 0,
        ExprData::Rational(r) => r.0 == 0,
        _ => false,
    }
}

pub(super) fn literal(pool: &ExprPool, e: ExprId) -> Option<Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

pub(super) fn rational_expr(pool: &ExprPool, q: &Rational) -> ExprId {
    if *q.denom() == 1 {
        pool.integer(q.numer().clone())
    } else {
        pool.rational(q.numer().clone(), q.denom().clone())
    }
}

/// What a closed-form attempt concluded.
pub(super) enum Outcome {
    /// The exact value, with the hypotheses it was derived under.
    Sum {
        value: ExprId,
        conditions: Vec<SideCondition>,
        rule: &'static str,
    },
    /// The series provably diverges (or a term is undefined); the message
    /// says why.
    Refused(String),
    /// Outside this module's reach: no claim either way.
    NotApplicable,
}

// ---------------------------------------------------------------------------
// Structural helpers
// ---------------------------------------------------------------------------

/// `e = A·k + B` with `A`, `B` free of `k` (unsimplified).
pub(super) fn affine_in_k(e: ExprId, k: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    if e == k {
        return Some((pool.integer(1_i32), pool.integer(0_i32)));
    }
    if is_free_of_k(e, k, pool) {
        return Some((pool.integer(0_i32), e));
    }
    match pool.get(e) {
        ExprData::Add(xs) => {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for &x in xs.iter() {
                let (ax, bx) = affine_in_k(x, k, pool)?;
                a.push(ax);
                b.push(bx);
            }
            Some((pool.add(a), pool.add(b)))
        }
        ExprData::Mul(xs) => {
            let mut inner = None;
            let mut rest = Vec::new();
            for &x in xs.iter() {
                if is_free_of_k(x, k, pool) {
                    rest.push(x);
                } else if inner.is_some() {
                    return None;
                } else {
                    inner = Some(affine_in_k(x, k, pool)?);
                }
            }
            let (a, b) = inner?;
            let scale = |v: ExprId| {
                let mut fs = rest.clone();
                fs.push(v);
                pool.mul(fs)
            };
            Some((scale(a), scale(b)))
        }
        _ => None,
    }
}

/// Coefficients `[c_0, …, c_d]` (simplified, trailing zeros trimmed) of `e`
/// as a polynomial in `k` whose coefficients are free of `k`.
pub(super) fn poly_coeffs(e: ExprId, k: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
    let raw = poly_coeffs_raw(e, k, pool, 0)?;
    let mut out: Vec<ExprId> = raw.into_iter().map(|c| simp(pool, c)).collect();
    while out.len() > 1 && is_zero(pool, *out.last()?) {
        out.pop();
    }
    Some(out)
}

pub(super) fn poly_mul(a: &[ExprId], b: &[ExprId], pool: &ExprPool) -> Option<Vec<ExprId>> {
    if a.len() + b.len() - 1 > MAX_DEGREE + 1 {
        return None;
    }
    let mut out: Vec<Vec<ExprId>> = vec![Vec::new(); a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j].push(pool.mul(vec![x, y]));
        }
    }
    Some(
        out.into_iter()
            .map(|terms| simp(pool, pool.add(terms)))
            .collect(),
    )
}

fn poly_coeffs_raw(e: ExprId, k: ExprId, pool: &ExprPool, depth: u32) -> Option<Vec<ExprId>> {
    if depth > 64 {
        return None;
    }
    if e == k {
        return Some(vec![pool.integer(0_i32), pool.integer(1_i32)]);
    }
    if is_free_of_k(e, k, pool) {
        return Some(vec![e]);
    }
    match pool.get(e) {
        ExprData::Add(xs) => {
            let mut acc: Vec<Vec<ExprId>> = Vec::new();
            for &x in xs.iter() {
                let c = poly_coeffs_raw(x, k, pool, depth + 1)?;
                if c.len() > acc.len() {
                    acc.resize(c.len(), Vec::new());
                }
                for (i, ci) in c.into_iter().enumerate() {
                    acc[i].push(ci);
                }
            }
            Some(acc.into_iter().map(|t| simp(pool, pool.add(t))).collect())
        }
        ExprData::Mul(xs) => {
            let mut acc = vec![pool.integer(1_i32)];
            for &x in xs.iter() {
                let c = poly_coeffs_raw(x, k, pool, depth + 1)?;
                acc = poly_mul(&acc, &c, pool)?;
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let n = match pool.get(exp) {
                ExprData::Integer(n) => n.0.to_usize()?,
                _ => return None,
            };
            if n > MAX_DEGREE {
                return None;
            }
            let b = poly_coeffs_raw(base, k, pool, depth + 1)?;
            let mut acc = vec![pool.integer(1_i32)];
            for _ in 0..n {
                acc = poly_mul(&acc, &b, pool)?;
            }
            Some(acc)
        }
        _ => None,
    }
}

pub(super) fn eval_poly(coeffs: &[ExprId], x: ExprId, pool: &ExprPool) -> ExprId {
    let mut terms = Vec::new();
    for (i, &c) in coeffs.iter().enumerate() {
        if is_zero(pool, c) {
            continue;
        }
        terms.push(match i {
            0 => c,
            1 => pool.mul(vec![c, x]),
            _ => pool.mul(vec![c, pool.pow(x, pool.integer(i as i64))]),
        });
    }
    if terms.is_empty() {
        pool.integer(0_i32)
    } else {
        pool.add(terms)
    }
}

fn binomial(n: usize, r: usize) -> Integer {
    let mut acc = Integer::from(1);
    for i in 0..r {
        acc *= n - i;
        acc /= i + 1;
    }
    acc
}

fn has_complex_symbol(e: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Symbol { domain, .. } => domain == Domain::Complex,
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&x| has_complex_symbol(x, pool)),
        ExprData::Pow { base, exp } => {
            has_complex_symbol(base, pool) || has_complex_symbol(exp, pool)
        }
        ExprData::Func { args, .. } => args.iter().any(|&x| has_complex_symbol(x, pool)),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Family 1: polynomial × geometric
// ---------------------------------------------------------------------------

/// One `P(k)·q^k` group of the summand.
struct GeoGroup {
    coeffs: Vec<ExprId>,
    q: ExprId,
}

/// Split `term` into `Σ_g P_g(k)·q_g^k`, or `None` when some additive piece is
/// not polynomial × geometric. A piece with no geometric factor (`q = 1`) is
/// also `None`: polynomial and rational sums belong to other paths.
fn geo_decompose(term: ExprId, k: ExprId, pool: &ExprPool) -> Option<Vec<GeoGroup>> {
    let pieces: Vec<ExprId> = match pool.get(term) {
        ExprData::Add(xs) => xs.to_vec(),
        _ => vec![term],
    };
    let mut groups: Vec<GeoGroup> = Vec::new();
    for piece in pieces {
        let factors: Vec<ExprId> = match pool.get(piece) {
            ExprData::Mul(xs) => xs.to_vec(),
            _ => vec![piece],
        };
        let mut q_parts = Vec::new();
        let mut coeff_parts = Vec::new();
        let mut poly_parts = Vec::new();
        for f in factors {
            if is_free_of_k(f, k, pool) {
                coeff_parts.push(f);
                continue;
            }
            match pool.get(f) {
                ExprData::Pow { base, exp }
                    if is_free_of_k(base, k, pool) && !is_free_of_k(exp, k, pool) =>
                {
                    let (a, b) = affine_in_k(exp, k, pool)?;
                    q_parts.push(pool.pow(base, a));
                    coeff_parts.push(pool.pow(base, b));
                }
                ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                    let (a, b) = affine_in_k(args[0], k, pool)?;
                    q_parts.push(pool.func("exp", vec![a]));
                    coeff_parts.push(pool.func("exp", vec![b]));
                }
                _ => poly_parts.push(f),
            }
        }
        if q_parts.is_empty() {
            return None;
        }
        let q = simp(pool, pool.mul(q_parts));
        if q == pool.integer(1_i32) || is_zero(pool, q) {
            return None;
        }
        let mut p = poly_coeffs(pool.mul(poly_parts), k, pool)?;
        let c = simp(pool, pool.mul(coeff_parts));
        for x in p.iter_mut() {
            *x = simp(pool, pool.mul(vec![c, *x]));
        }
        if let Some(g) = groups.iter_mut().find(|g| g.q == q) {
            if p.len() > g.coeffs.len() {
                g.coeffs.resize(p.len(), pool.integer(0_i32));
            }
            for (i, x) in p.into_iter().enumerate() {
                g.coeffs[i] = simp(pool, pool.add(vec![g.coeffs[i], x]));
            }
        } else {
            groups.push(GeoGroup { coeffs: p, q });
        }
    }
    groups.retain(|g| g.coeffs.iter().any(|&c| !is_zero(pool, c)));
    Some(groups)
}

/// Coefficients of `Q` with `q·Q(k+1) − Q(k) = P(k)`; `deg Q = deg P`.
///
/// Matching `k^i`: `(q − 1)·a_i + q·Σ_{j>i} C(j,i)·a_j = p_i`, solved from the
/// top degree down, dividing only by `q − 1`.
pub(super) fn geometric_antidifference(p: &[ExprId], q: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let d = p.len() - 1;
    let inv = pool.pow(
        pool.add(vec![q, pool.integer(-1_i32)]),
        pool.integer(-1_i32),
    );
    let mut a = vec![pool.integer(0_i32); d + 1];
    for i in (0..=d).rev() {
        let mut terms = vec![p[i]];
        for (j, &aj) in a.iter().enumerate().skip(i + 1) {
            terms.push(pool.mul(vec![pool.integer(-binomial(j, i)), q, aj]));
        }
        a[i] = simp(pool, pool.mul(vec![pool.add(terms), inv]));
    }
    a
}

/// `Q(x)·q^x`.
pub(super) fn antidifference_at(a: &[ExprId], q: ExprId, x: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![eval_poly(a, x, pool), pool.pow(q, x)])
}

/// Is `|q| < 1`? `Ok(None)` when established, `Ok(Some(cond))` when it has to
/// be assumed (the condition is what to assume), `Err(why)` when it fails.
pub(super) fn geometric_convergence(
    q: ExprId,
    pool: &ExprPool,
) -> Result<Option<SideCondition>, String> {
    let one = pool.integer(1_i32);
    let fail = |what: &str| {
        Err(format!(
            "the series diverges: its terms grow like {}^k and |{}| ≥ 1 ({what})",
            pool.display(q),
            pool.display(q)
        ))
    };
    if let Some(v) = literal(pool, q) {
        let mut abs = v;
        abs.abs_mut();
        return if abs < 1 { Ok(None) } else { fail("a number") };
    }
    let complex = has_complex_symbol(q, pool);
    // q = exp(z): |q| < 1 ⇔ z < 0 for a real z.
    if let ExprData::Func { name, args } = pool.get(q) {
        if name == "exp" && args.len() == 1 && !complex {
            let z = args[0];
            return match assumed_sign(z, pool) {
                Some(Sign::Negative) => Ok(None),
                Some(_) => fail("the exponent is not negative"),
                None => Ok(Some(SideCondition::Positive(simp(pool, neg(pool, z))))),
            };
        }
    }
    // q = b^e with a positive literal b ≠ 1: |q| < 1 ⇔ e·log b < 0.
    if let ExprData::Pow { base, exp } = pool.get(q) {
        if let Some(b) = literal(pool, base) {
            if b > 0 && b != 1 && !complex {
                let want = if b > 1 {
                    Sign::Negative
                } else {
                    Sign::Positive
                };
                return match assumed_sign(exp, pool) {
                    Some(s) if s == want => Ok(None),
                    Some(_) => fail("the exponent has the wrong sign"),
                    None => {
                        let gap = if b > 1 {
                            simp(pool, neg(pool, exp))
                        } else {
                            exp
                        };
                        Ok(Some(SideCondition::Positive(gap)))
                    }
                };
            }
        }
    }
    if !complex {
        let one_minus = simp(pool, pool.add(vec![one, neg(pool, q)]));
        let one_plus = simp(pool, pool.add(vec![one, q]));
        let below = assumed_sign(one_minus, pool);
        let above = assumed_sign(one_plus, pool);
        if matches!(below, Some(Sign::Negative | Sign::Zero)) {
            return fail("it is at least 1");
        }
        if matches!(above, Some(Sign::Negative | Sign::Zero)) {
            return fail("it is at most −1");
        }
        match (below, above) {
            (Some(Sign::Positive), Some(Sign::Positive)) => return Ok(None),
            // Known `q > −1`: only `q < 1` is left to assume.
            (None, Some(Sign::Positive)) => return Ok(Some(SideCondition::Positive(one_minus))),
            // Known `q < 1`: only `q > −1` is left.
            (Some(Sign::Positive), None) => return Ok(Some(SideCondition::Positive(one_plus))),
            _ => {}
        }
    }
    let gap = simp(
        pool,
        pool.add(vec![one, neg(pool, pool.func("abs", vec![q]))]),
    );
    Ok(Some(SideCondition::Positive(gap)))
}

/// The triangular solve leaves nested `(q − 1)^{−j}` terms; over a common
/// denominator (`r·(1 + r)/(1 − r)^3` for `Σ k²rᵏ`) is usually far shorter.
/// Kept only when it is shorter — both forms are the same rational function.
pub(super) fn tidy(value: ExprId, pool: &ExprPool) -> ExprId {
    let Ok(c) = crate::poly::cancel(value, vec![], pool) else {
        return value;
    };
    let len = |e: ExprId| pool.display(e).to_string().len();
    if len(c) < len(value) {
        c
    } else {
        value
    }
}

/// `Σ_{k=lo}^{hi} P(k)·q^k` for finite bounds (the `q = 1` branch is a
/// recorded `q − 1 ≠ 0` hypothesis per group).
pub(super) fn finite_geometric(
    term: ExprId,
    k: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Outcome {
    let Some(groups) = geo_decompose(simp(pool, term), k, pool) else {
        return Outcome::NotApplicable;
    };
    if groups.is_empty() {
        return Outcome::NotApplicable;
    }
    let hi_p1 = simp(pool, pool.add(vec![hi, pool.integer(1_i32)]));
    let mut parts = Vec::new();
    let mut conditions = Vec::new();
    for g in &groups {
        let a = geometric_antidifference(&g.coeffs, g.q, pool);
        parts.push(antidifference_at(&a, g.q, hi_p1, pool));
        parts.push(neg(pool, antidifference_at(&a, g.q, lo, pool)));
        if literal(pool, g.q).is_none() {
            let cond =
                SideCondition::NonZero(simp(pool, pool.add(vec![g.q, pool.integer(-1_i32)])));
            if !conditions.contains(&cond) {
                conditions.push(cond);
            }
        }
    }
    Outcome::Sum {
        value: tidy(simp(pool, pool.add(parts)), pool),
        conditions,
        rule: if groups.iter().all(|g| g.coeffs.len() == 1) {
            "geometric_symbolic_ratio"
        } else {
            "arithmetico_geometric"
        },
    }
}

/// `Σ_{k=lo}^{∞} P(k)·q^k = −Q(lo)·q^{lo}` under `|q| < 1`.
fn infinite_geometric(term: ExprId, k: ExprId, lo: ExprId, pool: &ExprPool) -> Outcome {
    let Some(groups) = geo_decompose(term, k, pool) else {
        return Outcome::NotApplicable;
    };
    if groups.is_empty() {
        return Outcome::NotApplicable;
    }
    let mut parts = Vec::new();
    let mut conditions = Vec::new();
    for g in &groups {
        match geometric_convergence(g.q, pool) {
            Err(why) => return Outcome::Refused(why),
            Ok(Some(c)) => {
                if !conditions.contains(&c) {
                    conditions.push(c);
                }
            }
            Ok(None) => {}
        }
        let a = geometric_antidifference(&g.coeffs, g.q, pool);
        parts.push(neg(pool, antidifference_at(&a, g.q, lo, pool)));
    }
    Outcome::Sum {
        value: tidy(simp(pool, pool.add(parts)), pool),
        conditions,
        rule: if groups.iter().all(|g| g.coeffs.len() == 1) {
            "geometric_series_infinite"
        } else {
            "arithmetico_geometric_infinite"
        },
    }
}

// ---------------------------------------------------------------------------
// Family 2: rational series to ∞
// ---------------------------------------------------------------------------

/// `R(k) = scale · N(k) · Π (k − ρ_j)^{−m_j}` for one additive piece.
pub(super) struct RationalPiece {
    pub(super) numer: Vec<ExprId>,
    pub(super) poles: Vec<(ExprId, u32)>,
    /// Simple irreducible quadratic factors `k² + p·k + s` of the
    /// denominator, as `(p, s)` with `p` a rational literal and `s` free of
    /// `k` (F12). Their poles `α ± iβ` (`α = −p/2`, `β² = s − p²/4`) are
    /// complex, so they never telescope against a real pole.
    pub(super) quads: Vec<(Rational, ExprId)>,
}

/// Factorisation over ℚ of a polynomial with rational literal coefficients:
/// `(linear roots with multiplicity, monic irreducible quadratics (p, s,
/// multiplicity), leading coefficient)`, or `None` when some irreducible
/// factor has degree > 2.
fn rational_factor_roots(
    coeffs: &[Rational],
) -> Option<(
    Vec<(Rational, u32)>,
    Vec<(Rational, Rational, u32)>,
    Rational,
)> {
    let lc = coeffs.last()?.clone();
    if lc == 0 {
        return None;
    }
    // Clear denominators into a ℤ polynomial and factor it.
    let mut den = Integer::from(1);
    for c in coeffs {
        den = den.lcm(c.denom());
    }
    let ints: Vec<Integer> = coeffs
        .iter()
        .map(|c| (c.clone() * Rational::from(den.clone())).numer().clone())
        .collect();
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let terms: Vec<ExprId> = ints
        .iter()
        .enumerate()
        .map(|(i, c)| {
            pool.mul(vec![
                pool.integer(c.clone()),
                pool.pow(x, pool.integer(i as i64)),
            ])
        })
        .collect();
    let p = crate::poly::UniPoly::from_symbolic(pool.add(terms), x, &pool).ok()?;
    let fac = p.factor_z().ok()?;
    let mut roots = Vec::new();
    let mut quads = Vec::new();
    for (f, m) in &fac.factors {
        let c = f.coefficients();
        match f.degree() {
            1 => {
                let root = Rational::from((-c[0].clone(), c[1].clone()));
                roots.push((root, *m));
            }
            2 => {
                let p = Rational::from((c[1].clone(), c[2].clone()));
                let s = Rational::from((c[0].clone(), c[2].clone()));
                quads.push((p, s, *m));
            }
            _ => return None,
        }
    }
    Some((roots, quads, lc))
}

pub(super) fn rational_piece(piece: ExprId, k: ExprId, pool: &ExprPool) -> Option<RationalPiece> {
    let mut work: Vec<ExprId> = match pool.get(piece) {
        ExprData::Mul(xs) => xs.to_vec(),
        _ => vec![piece],
    };
    let mut scale = Vec::new();
    let mut numer = Vec::new();
    let mut poles: Vec<(ExprId, u32)> = Vec::new();
    let mut quads: Vec<(Rational, ExprId)> = Vec::new();
    let mut found_pole = false;
    let mut steps = 0_u32;
    while let Some(f) = work.pop() {
        steps += 1;
        if steps > 256 {
            return None;
        }
        if is_free_of_k(f, k, pool) {
            scale.push(f);
            continue;
        }
        let neg_pow = match pool.get(f) {
            ExprData::Pow { base, exp } => match pool.get(exp) {
                ExprData::Integer(n) if n.0 < 0 => Some((base, (-n.0.clone()).to_u32()?)),
                _ => None,
            },
            _ => None,
        };
        let Some((base, m)) = neg_pow else {
            numer.push(f);
            continue;
        };
        // `(f₁·f₂)^{−m}` and `(g^e)^{−m}`: one pole factor at a time, so
        // `1/((k+a)(k+b))` is read as two linear poles rather than as a
        // quadratic with symbolic coefficients.
        match pool.get(base) {
            ExprData::Mul(xs) => {
                for &x in xs.iter() {
                    work.push(pool.pow(x, pool.integer(-(m as i64))));
                }
                continue;
            }
            ExprData::Pow { base: b2, exp: e2 } => {
                if let ExprData::Integer(e) = pool.get(e2) {
                    if e.0 > 0 {
                        let e = e.0.to_u32()?;
                        work.push(pool.pow(b2, pool.integer(-((m.checked_mul(e)?) as i64))));
                        continue;
                    }
                }
            }
            _ => {}
        }
        found_pole = true;
        let bc = poly_coeffs(base, k, pool)?;
        if bc.len() < 2 {
            return None;
        }
        if bc.len() == 2 {
            // `A·k + B` with a literal `A ≠ 0` and any `B` free of `k`.
            let a = literal(pool, bc[1])?;
            if a == 0 {
                return None;
            }
            let rho = simp(
                pool,
                pool.mul(vec![
                    neg(pool, bc[0]),
                    rational_expr(pool, &(Rational::from(1) / a.clone())),
                ]),
            );
            scale.push(pool.pow(rational_expr(pool, &a), pool.integer(-(m as i64))));
            poles.push((rho, m));
            continue;
        }
        let lits: Option<Vec<Rational>> = bc.iter().map(|&c| literal(pool, c)).collect();
        let Some(lits) = lits else {
            // `k² + p·k + s` with literal `p` and a symbolic `s` (`k² + a²`):
            // an irreducible quadratic when `s − p²/4 > 0`, which the
            // quadratic path decides or states.
            if bc.len() != 3 || m != 1 {
                return None;
            }
            let a2 = literal(pool, bc[2])?;
            let a1 = literal(pool, bc[1])?;
            if a2 == 0 {
                return None;
            }
            let inv = rational_expr(pool, &(Rational::from(1) / a2.clone()));
            scale.push(inv);
            quads.push((a1 / a2, simp(pool, pool.mul(vec![bc[0], inv]))));
            continue;
        };
        let (roots, qs, lc) = rational_factor_roots(&lits)?;
        scale.push(pool.pow(rational_expr(pool, &lc), pool.integer(-(m as i64))));
        for (r, mult) in roots {
            poles.push((rational_expr(pool, &r), mult.checked_mul(m)?));
        }
        for (p, s, mult) in qs {
            if mult.checked_mul(m)? != 1 {
                return None;
            }
            quads.push((p, rational_expr(pool, &s)));
        }
    }
    if !found_pole {
        return None;
    }
    let mut n = poly_coeffs(pool.mul(numer), k, pool)?;
    let s = simp(pool, pool.mul(scale));
    for x in n.iter_mut() {
        *x = simp(pool, pool.mul(vec![s, *x]));
    }
    // Merge coincident poles.
    let mut merged: Vec<(ExprId, u32)> = Vec::new();
    for (rho, m) in poles {
        if let Some(e) = merged.iter_mut().find(|(r, _)| *r == rho) {
            e.1 += m;
        } else {
            merged.push((rho, m));
        }
    }
    Some(RationalPiece {
        numer: n,
        poles: merged,
        quads,
    })
}

/// Partial-fraction coefficients `c_{ρ,s}` (`s = 1..=m`) of one piece, keyed
/// by pole; `None` when the piece has a polynomial part (`deg N ≥ Σ m`), which
/// a convergent series cannot have.
pub(super) fn partial_fractions(
    piece: &RationalPiece,
    k: ExprId,
    pool: &ExprPool,
    conditions: &mut Vec<SideCondition>,
) -> Option<Vec<(ExprId, Vec<ExprId>)>> {
    let total: u32 = piece.poles.iter().map(|p| p.1).sum::<u32>() + 2 * piece.quads.len() as u32;
    if piece.numer.len() as u32 > total {
        return None;
    }
    // A repeated quadratic factor would need higher-order complex poles.
    for (i, qi) in piece.quads.iter().enumerate() {
        if piece.quads.iter().skip(i + 1).any(|qj| qj == qi) {
            return None;
        }
    }
    // Distinct poles must stay distinct: their differences are divisors.
    for (i, &(ri, _)) in piece.poles.iter().enumerate() {
        for &(rj, _) in piece.poles.iter().skip(i + 1) {
            let d = simp(pool, pool.add(vec![ri, neg(pool, rj)]));
            if provably_zero(pool, d) {
                return None;
            }
            if literal(pool, d).is_none() {
                let c = SideCondition::NonZero(d);
                if !conditions.contains(&c) {
                    conditions.push(c);
                }
            }
        }
    }
    let numer = eval_poly(&piece.numer, k, pool);
    let mut out = Vec::new();
    for (j, &(rho, m)) in piece.poles.iter().enumerate() {
        // g_j(k) = N(k) · Π_{i≠j} (k − ρ_i)^{−m_i}
        let mut fs = vec![numer];
        for (i, &(ri, mi)) in piece.poles.iter().enumerate() {
            if i != j {
                fs.push(pool.pow(pool.add(vec![k, neg(pool, ri)]), pool.integer(-(mi as i64))));
            }
        }
        for (p, s) in &piece.quads {
            fs.push(pool.pow(quad_expr(p, *s, k, pool), pool.integer(-1_i32)));
        }
        let mut g = pool.mul(fs);
        let mut at = HashMap::new();
        at.insert(k, rho);
        let mut coeffs = vec![pool.integer(0_i32); m as usize];
        let mut fact = Integer::from(1);
        for t in 0..m {
            if t > 0 {
                fact *= t;
                g = crate::diff::diff(g, k, pool).ok()?.value;
            }
            let v = simp(pool, subs(g, &at, pool));
            let c = simp(
                pool,
                pool.mul(vec![v, pool.rational(Integer::from(1), fact.clone())]),
            );
            // The coefficient of (k − ρ)^{−(m − t)}.
            coeffs[(m - t - 1) as usize] = c;
        }
        out.push((rho, coeffs));
    }
    Some(out)
}

/// `k² + p·k + s`.
fn quad_expr(p: &Rational, s: ExprId, k: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![
        pool.pow(k, pool.integer(2_i32)),
        pool.mul(vec![rational_expr(pool, p), k]),
        s,
    ])
}

/// Reduce a coefficient vector modulo the monic `k² + p·k + s`, to `[c₀, c₁]`.
fn reduce_mod_quad(c: &[ExprId], p: &Rational, s: ExprId, pool: &ExprPool) -> [ExprId; 2] {
    let mut c: Vec<ExprId> = c.to_vec();
    let pe = rational_expr(pool, p);
    for i in (2..c.len()).rev() {
        let top = c[i];
        c[i - 1] = simp(
            pool,
            pool.add(vec![c[i - 1], neg(pool, pool.mul(vec![pe, top]))]),
        );
        c[i - 2] = simp(
            pool,
            pool.add(vec![c[i - 2], neg(pool, pool.mul(vec![s, top]))]),
        );
    }
    let zero = pool.integer(0_i32);
    [
        c.first().copied().unwrap_or(zero),
        c.get(1).copied().unwrap_or(zero),
    ]
}

/// The numerators `A·k + B` of the quadratic partial fractions of `piece`:
/// `piece = Σ_ρ … + Σ_Q (A_Q·k + B_Q)/Q(k)`. With `L` the rest of the
/// denominator, `A·k + B ≡ N·L^{−1} (mod Q)`; the inverse of a linear
/// `l₁k + l₀` modulo `Q` is `(−l₁k + l₀ − p·l₁)/(l₁²s − p·l₀l₁ + l₀²)`.
fn quadratic_numerators(
    piece: &RationalPiece,
    pool: &ExprPool,
) -> Option<Vec<(Rational, ExprId, ExprId, ExprId)>> {
    let one = pool.integer(1_i32);
    let mut out = Vec::new();
    for (qi, (p, s)) in piece.quads.iter().enumerate() {
        let mut l = vec![one];
        for &(rho, m) in &piece.poles {
            for _ in 0..m {
                l = poly_mul(&l, &[neg(pool, rho), one], pool)?;
            }
        }
        for (qj, (p2, s2)) in piece.quads.iter().enumerate() {
            if qj != qi {
                l = poly_mul(&l, &[*s2, rational_expr(pool, p2), one], pool)?;
            }
        }
        let [n0, n1] = reduce_mod_quad(&piece.numer, p, *s, pool);
        let [l0, l1] = reduce_mod_quad(&l, p, *s, pool);
        let pe = rational_expr(pool, p);
        let norm = simp(
            pool,
            pool.add(vec![
                pool.mul(vec![l1, l1, *s]),
                neg(pool, pool.mul(vec![pe, l0, l1])),
                pool.mul(vec![l0, l0]),
            ]),
        );
        if provably_zero(pool, norm) {
            return None;
        }
        let m0 = pool.add(vec![l0, neg(pool, pool.mul(vec![pe, l1]))]);
        let inv = pool.pow(norm, pool.integer(-1_i32));
        let a = pool.add(vec![
            pool.mul(vec![n1, m0]),
            neg(pool, pool.mul(vec![n0, l1])),
            pool.mul(vec![n1, l1, pe]),
        ]);
        let b = pool.add(vec![pool.mul(vec![n0, m0]), pool.mul(vec![n1, l1, *s])]);
        out.push((
            p.clone(),
            *s,
            tidy(simp(pool, pool.mul(vec![a, inv])), pool),
            tidy(simp(pool, pool.mul(vec![b, inv])), pool),
        ));
    }
    Some(out)
}

/// `Im ψ(x + iβ)` for a literal `x` that is an integer or a half-integer
/// (DLMF 5.4.17, 5.4.18 and the recurrence `ψ(z + 1) = ψ(z) + 1/z`):
/// `Im ψ(iβ) = 1/(2β) + (π/2)·coth(πβ)`, `Im ψ(½ + iβ) = (π/2)·tanh(πβ)`,
/// and each unit step adds `Im 1/(u + iβ) = −β/(u² + β²)`.
fn im_digamma(x: &Rational, beta: ExprId, beta2: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let half = Rational::from((1, 2));
    let f = if *x.denom() == 1 {
        Rational::new()
    } else if *x.denom() == 2 {
        half.clone()
    } else {
        return None;
    };
    let steps = (x.clone() - f.clone()).numer().to_i64()?;
    if steps.abs() > MAX_TELESCOPE_TERMS {
        return None;
    }
    let pi = pool.symbol("pi", Domain::Real);
    let tanh = pool.func("tanh", vec![pool.mul(vec![pi, beta])]);
    let mut parts = if f == 0 {
        vec![
            pool.mul(vec![
                pool.rational(1, 2),
                pool.pow(beta, pool.integer(-1_i32)),
            ]),
            pool.mul(vec![
                pool.rational(1, 2),
                pi,
                pool.pow(tanh, pool.integer(-1_i32)),
            ]),
        ]
    } else {
        vec![pool.mul(vec![pool.rational(1, 2), pi, tanh])]
    };
    // β/(u² + β²) for u = f + j over the steps between f and x.
    let (from, to, sign) = if steps >= 0 {
        (0, steps, -1)
    } else {
        (steps, 0, 1)
    };
    for j in from..to {
        if crate::budget::check().is_err() {
            return None;
        }
        let u = f.clone() + Rational::from(j);
        let u2 = rational_expr(pool, &(u.clone() * u));
        parts.push(pool.mul(vec![
            pool.integer(sign),
            beta,
            pool.pow(pool.add(vec![u2, beta2]), pool.integer(-1_i32)),
        ]));
    }
    Some(pool.add(parts))
}

/// `β` with `β² = beta2`: the base of a literal square (`a²` gives `a`; the
/// closed forms below are even in `β`), else `√beta2`.
fn sqrt_of_square(beta2: ExprId, pool: &ExprPool) -> ExprId {
    if let ExprData::Pow { base, exp } = pool.get(beta2) {
        if matches!(pool.get(exp), ExprData::Integer(n) if n.0 == 2) {
            return base;
        }
    }
    simp(pool, pool.pow(beta2, pool.rational(1, 2)))
}

/// The contribution of the irreducible quadratic partial fractions
/// `Σ_{k≥lo} (A·k + B)/((k − α)² + β²)` (regularised: their `1/k` tails are
/// accounted for by the caller's convergence check).
///
/// Writing the pair of conjugate poles `α ± iβ`, the sum is
/// `(C/β)·Im ψ(x₀ + iβ) − A·Re ψ(x₀ + iβ)` with `x₀ = lo − α`,
/// `C = B + A·α`. `Im ψ` is elementary at integer and half-integer `x₀`;
/// `Re ψ(x₀ + iβ)` is not, so the `A` parts must cancel within each class of
/// poles sharing `β²` and differing by an integer in `α` — then only finitely
/// many `Re 1/(u + iβ) = u/(u² + β²)` terms remain. `None` when they do not.
fn quadratic_contribution(
    quads: &[(Rational, ExprId, ExprId, ExprId)],
    lo: &Rational,
    pool: &ExprPool,
    conditions: &mut Vec<SideCondition>,
) -> Option<ExprId> {
    let mut parts = Vec::new();
    // (β², frac(x₀)) → members (x₀, A)
    let mut classes: Vec<(ExprId, Rational, Vec<(Rational, ExprId)>)> = Vec::new();
    for (p, beta2_raw, a, b) in quads {
        let alpha = -p.clone() / Rational::from(2);
        let beta2 = simp(
            pool,
            pool.add(vec![
                *beta2_raw,
                rational_expr(pool, &(-(alpha.clone() * alpha.clone()))),
            ]),
        );
        match literal(pool, beta2) {
            Some(v) if v > 0 => {}
            Some(_) => return None,
            None => match assumed_sign(beta2, pool) {
                Some(Sign::Positive) => {}
                Some(_) => return None,
                None => {
                    let c = SideCondition::Positive(beta2);
                    if !conditions.contains(&c) {
                        conditions.push(c);
                    }
                }
            },
        }
        let beta = sqrt_of_square(beta2, pool);
        let x0 = lo.clone() - alpha.clone();
        let c = simp(
            pool,
            pool.add(vec![*b, pool.mul(vec![*a, rational_expr(pool, &alpha)])]),
        );
        if !provably_zero(pool, c) {
            let im = im_digamma(&x0, beta, beta2, pool)?;
            parts.push(pool.mul(vec![c, pool.pow(beta, pool.integer(-1_i32)), im]));
        }
        if !provably_zero(pool, *a) {
            let frac = x0.clone() - Rational::from(x0.clone().floor());
            match classes
                .iter_mut()
                .find(|(b2, f, _)| *b2 == beta2 && *f == frac)
            {
                Some(cl) => cl.2.push((x0, *a)),
                None => classes.push((beta2, frac, vec![(x0, *a)])),
            }
        }
    }
    for (beta2, _, members) in classes {
        let total = simp(
            pool,
            pool.add(members.iter().map(|m| m.1).collect::<Vec<_>>()),
        );
        if !provably_zero(pool, total) {
            return None;
        }
        let xmin = members.iter().map(|m| m.0.clone()).min()?;
        for (x, a) in &members {
            let d = (x.clone() - xmin.clone()).numer().to_i64()?;
            if d > MAX_TELESCOPE_TERMS {
                return None;
            }
            for t in 0..d {
                if crate::budget::check().is_err() {
                    return None;
                }
                let u = xmin.clone() + Rational::from(t);
                let u2 = rational_expr(pool, &(u.clone() * u.clone()));
                // −A·Re 1/(u + iβ) = −A·u/(u² + β²)
                parts.push(pool.mul(vec![
                    neg(pool, *a),
                    rational_expr(pool, &u),
                    pool.pow(pool.add(vec![u2, beta2]), pool.integer(-1_i32)),
                ]));
            }
        }
    }
    Some(pool.add(parts))
}

/// `ζ(s, x) = Σ_{i≥0} (x + i)^{−s}` for `s ≥ 2`: a closed form at integer and
/// half-integer `x` (`π^s` multiples for even `s`, `zeta(s)` for odd `s`),
/// `trigamma` for `s = 2`, else the `hurwitz_zeta(s, x)` node; `None` only
/// for a non-positive literal start.
fn hurwitz_zeta(s: usize, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let pi = pool.symbol("pi", Domain::Real);
    if let Some(v) = literal(pool, x) {
        let half = Rational::from((1, 2));
        let (base, start): (Rational, Rational) = if *v.denom() == 1 {
            (Rational::from(1), Rational::from(1))
        } else if *v.denom() == 2 {
            (half.clone(), half.clone())
        } else {
            return if s == 2 {
                Some(pool.func("trigamma", vec![x]))
            } else {
                // ζ(s, x) at another rational point: the Hurwitz node.
                Some(pool.func("hurwitz_zeta", vec![pool.integer(s as i64), x]))
            };
        };
        if v <= 0 {
            return None;
        }
        // ζ(s, base) for base ∈ {1, 1/2}, then subtract the leading terms.
        // Even s: a rational multiple of π^s. Odd s: `zeta(s)` itself, which
        // has no known closed form (G11b) — named, never approximated.
        let (zc, zeta_s) = if s % 2 == 0 {
            (
                zeta_even_coefficient((s / 2) as u32),
                pool.pow(pi, pool.integer(s as i64)),
            )
        } else {
            (
                Rational::from(1),
                pool.func("zeta", vec![pool.integer(s as i64)]),
            )
        };
        let mult = if base == 1 {
            zc
        } else {
            zc * Rational::from((Integer::from(1) << s as u32) - 1)
        };
        let span = v - start.clone();
        if *span.denom() != 1 {
            return None;
        }
        let n_terms = span.numer().to_i64()?;
        if n_terms > MAX_TELESCOPE_TERMS {
            return None;
        }
        let mut corr = Rational::new();
        for i in 0..n_terms {
            if crate::budget::check().is_err() {
                return None;
            }
            let term = start.clone() + Rational::from(i);
            let mut p = Rational::from(1);
            for _ in 0..s {
                p *= term.clone();
            }
            corr += Rational::from(1) / p;
        }
        let zeta = pool.mul(vec![rational_expr(pool, &mult), zeta_s]);
        return Some(pool.add(vec![zeta, rational_expr(pool, &(-corr))]));
    }
    if s == 2 {
        return Some(pool.func("trigamma", vec![x]));
    }
    Some(pool.func("hurwitz_zeta", vec![pool.integer(s as i64), x]))
}

/// `Σ_{k=lo}^{∞} R(k)` for a rational `R` whose poles are linear in `k`.
fn infinite_rational(term: ExprId, k: ExprId, lo: ExprId, pool: &ExprPool) -> Outcome {
    let pieces: Vec<ExprId> = match pool.get(term) {
        ExprData::Add(xs) => xs.to_vec(),
        _ => vec![term],
    };
    let mut conditions: Vec<SideCondition> = Vec::new();
    // pole → coefficient by order (index s−1)
    let mut poles: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
    // Irreducible quadratic factors: (p, s, A, B) of (A·k + B)/(k² + p·k + s).
    let mut quads: Vec<(Rational, ExprId, ExprId, ExprId)> = Vec::new();
    for piece in pieces {
        let Some(rp) = rational_piece(piece, k, pool) else {
            return Outcome::NotApplicable;
        };
        let Some(pf) = partial_fractions(&rp, k, pool, &mut conditions) else {
            return Outcome::NotApplicable;
        };
        if !rp.quads.is_empty() {
            let Some(qn) = quadratic_numerators(&rp, pool) else {
                return Outcome::NotApplicable;
            };
            for (p, s, a, b) in qn {
                match quads.iter_mut().find(|q| q.0 == p && q.1 == s) {
                    Some(q) => {
                        q.2 = simp(pool, pool.add(vec![q.2, a]));
                        q.3 = simp(pool, pool.add(vec![q.3, b]));
                    }
                    None => quads.push((p, s, a, b)),
                }
            }
        }
        for (rho, cs) in pf {
            let slot = match poles.iter().position(|(r, _)| *r == rho) {
                Some(i) => i,
                None => {
                    poles.push((rho, Vec::new()));
                    poles.len() - 1
                }
            };
            let entry = &mut poles[slot].1;
            if cs.len() > entry.len() {
                entry.resize(cs.len(), pool.integer(0_i32));
            }
            for (s, c) in cs.into_iter().enumerate() {
                entry[s] = simp(pool, pool.add(vec![entry[s], c]));
            }
        }
    }
    // Convergence: the 1/k coefficient of the tail, Σ c_{ρ,1} (+ Σ A for
    // the quadratic factors), must vanish.
    let s1 = simp(
        pool,
        pool.add(
            poles
                .iter()
                .map(|(_, cs)| cs[0])
                .chain(quads.iter().map(|q| q.2))
                .collect::<Vec<_>>(),
        ),
    );
    if !provably_zero(pool, s1) {
        if literal(pool, s1).is_some() {
            return Outcome::Refused(format!(
                "the series diverges: its terms decay like ({})/k, and Σ 1/k diverges",
                pool.display(s1)
            ));
        }
        return Outcome::NotApplicable;
    }
    // A pole at or after `lo` makes a term undefined.
    for (rho, _) in &poles {
        let x = simp(pool, pool.add(vec![lo, neg(pool, *rho)]));
        if let Some(v) = literal(pool, x) {
            if *v.denom() == 1 && v <= 0 {
                let at = simp(pool, *rho);
                return Outcome::Refused(format!(
                    "the summand is undefined at k = {}, which lies inside the summation range",
                    pool.display(at)
                ));
            }
        }
    }
    // Classes of poles differing by an integer.
    let mut classes: Vec<Vec<(i64, usize)>> = Vec::new();
    let mut reps: Vec<ExprId> = Vec::new();
    'pole: for (idx, (rho, _)) in poles.iter().enumerate() {
        for (ci, &rep) in reps.iter().enumerate() {
            let d = simp(pool, pool.add(vec![*rho, neg(pool, rep)]));
            if let Some(v) = literal(pool, d) {
                if *v.denom() == 1 {
                    let Some(off) = v.numer().to_i64() else {
                        return Outcome::NotApplicable;
                    };
                    if off.abs() > MAX_TELESCOPE_TERMS {
                        return Outcome::NotApplicable;
                    }
                    classes[ci].push((off, idx));
                    continue 'pole;
                }
            }
        }
        reps.push(*rho);
        classes.push(vec![(0, idx)]);
    }
    let mut parts = Vec::new();
    // `(Σ_class c_{ρ,1}, x0)` per class whose order-1 part does not telescope.
    let mut psi_parts: Vec<(ExprId, ExprId)> = Vec::new();
    for (ci, members) in classes.iter().enumerate() {
        let n_max = members.iter().map(|m| m.0).max().unwrap_or(0);
        // x0 = lo − ρ_rep − n_max: the smallest shift in the class.
        let x0 = simp(
            pool,
            pool.add(vec![lo, neg(pool, reps[ci]), pool.integer(-n_max)]),
        );
        match literal(pool, x0) {
            Some(_) => {}
            None => {
                if assumed_sign(x0, pool) != Some(Sign::Positive) {
                    let c = SideCondition::Positive(x0);
                    if !conditions.contains(&c) {
                        conditions.push(c);
                    }
                }
            }
        }
        let max_order = members
            .iter()
            .map(|&(_, i)| poles[i].1.len())
            .max()
            .unwrap_or(0);
        for s in 1..=max_order {
            let coeff_of = |i: usize| poles[i].1.get(s - 1).copied();
            // Telescoped remainder: Σ_j c_j Σ_{i<d_j} (x0+i)^{−s}, d_j = n_max − n_j.
            for &(off, i) in members {
                let Some(c) = coeff_of(i) else { continue };
                if is_zero(pool, c) {
                    continue;
                }
                let dj = n_max - off;
                let mut inner = Vec::new();
                for t in 0..dj {
                    if crate::budget::check().is_err() {
                        return Outcome::NotApplicable;
                    }
                    inner.push(pool.pow(
                        pool.add(vec![x0, pool.integer(t)]),
                        pool.integer(-(s as i64)),
                    ));
                }
                if inner.is_empty() {
                    continue;
                }
                // F_s(x0 + d) = F_s(x0) − Σ_{i<d} (x0+i)^{−s} for F_s = ζ(s, ·)
                // and for the regularised F_1 = −ψ alike.
                parts.push(pool.mul(vec![pool.integer(-1_i32), c, pool.add(inner)]));
            }
            let total = simp(
                pool,
                pool.add(
                    members
                        .iter()
                        .filter_map(|&(_, i)| coeff_of(i))
                        .collect::<Vec<_>>(),
                ),
            );
            if provably_zero(pool, total) {
                continue;
            }
            if s == 1 {
                psi_parts.push((total, x0));
                continue;
            }
            let special = hurwitz_zeta(s, x0, pool);
            let Some(sp) = special else {
                return Outcome::NotApplicable;
            };
            parts.push(pool.mul(vec![total, sp]));
        }
    }
    // Order 1: `−Σ total·ψ(x0)`. The totals sum to zero (convergence), so
    // Euler's γ inside every ψ cancels; at integer and half-integer points the
    // rest of ψ is exact (`H_{n−1}`, `−2·log 2 + Σ 2/(2j−1)`) and is used
    // instead — `ψ(1)` would otherwise fold to a float `−γ`.
    if !psi_parts.is_empty() {
        let exact: Option<Vec<ExprId>> = psi_parts
            .iter()
            .map(|&(total, x0)| {
                let r = digamma_without_gamma(x0, pool)?;
                Some(pool.mul(vec![pool.integer(-1_i32), total, r]))
            })
            .collect();
        match exact {
            Some(ps) => parts.extend(ps),
            None => {
                for &(total, x0) in &psi_parts {
                    parts.push(pool.mul(vec![
                        pool.integer(-1_i32),
                        total,
                        pool.func("digamma", vec![x0]),
                    ]));
                }
            }
        }
    }
    if !quads.is_empty() {
        // Complex poles: closed forms need a literal integer lower bound
        // (`x₀ = lo − α` an integer or half-integer).
        let Some(lo_v) = literal(pool, lo).filter(|v| *v.denom() == 1) else {
            return Outcome::NotApplicable;
        };
        let Some(q) = quadratic_contribution(&quads, &lo_v, pool, &mut conditions) else {
            return Outcome::NotApplicable;
        };
        parts.push(q);
    }
    let value = if parts.is_empty() {
        pool.integer(0_i32)
    } else {
        simp(pool, pool.add(parts))
    };
    // The coth/tanh terms of conjugate pole pairs often cancel
    // (`Σ k/(k⁴ + 4) = 3/8`); over ℚ(generators) that is visible.
    let value = if quads.is_empty() {
        value
    } else {
        tidy(value, pool)
    };
    if has_float(value, pool) && !has_float(term, pool) {
        // A special-function value folded to a float somewhere: that is not an
        // exact closed form, so this path has no answer.
        return Outcome::NotApplicable;
    }
    Outcome::Sum {
        value,
        conditions,
        rule: "rational_series_partial_fractions",
    }
}

/// `ψ(x) + γ` at a literal `x` that is a positive integer or a positive
/// half-integer: `H_{x−1}`, resp. `−2·log 2 + Σ_{j=1}^{n} 2/(2j−1)` for
/// `x = n + 1/2`. `None` elsewhere (or past [`MAX_TELESCOPE_TERMS`]).
fn digamma_without_gamma(x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let v = literal(pool, x)?;
    if v <= 0 {
        return None;
    }
    if *v.denom() == 1 {
        let n = v.numer().to_i64()?;
        if n > MAX_TELESCOPE_TERMS {
            return None;
        }
        let mut h = Rational::new();
        for j in 1..n {
            h += Rational::from((1, j));
        }
        return Some(rational_expr(pool, &h));
    }
    if *v.denom() == 2 {
        let n = (v - Rational::from((1, 2))).numer().to_i64()?;
        if n > MAX_TELESCOPE_TERMS {
            return None;
        }
        let mut h = Rational::new();
        for j in 1..=n {
            h += Rational::from((2, 2 * j - 1));
        }
        let log2 = pool.func("log", vec![pool.integer(2_i32)]);
        return Some(pool.add(vec![
            rational_expr(pool, &h),
            pool.mul(vec![pool.integer(-2_i32), log2]),
        ]));
    }
    None
}

pub(super) fn has_float(e: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Float(_) => true,
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&x| has_float(x, pool)),
        ExprData::Pow { base, exp } => has_float(base, pool) || has_float(exp, pool),
        ExprData::Func { args, .. } => args.iter().any(|&x| has_float(x, pool)),
        _ => false,
    }
}

/// `e = 0`, decided after simplification and, failing that, over ℚ(generators)
/// by [`crate::poly::cancel`] — so `1/(b−a) + 1/(a−b)` is zero. `false` means
/// *not established*.
pub(super) fn provably_zero(pool: &ExprPool, e: ExprId) -> bool {
    let e = simp(pool, e);
    if is_zero(pool, e) {
        return true;
    }
    match crate::poly::cancel(e, vec![], pool) {
        Ok(c) => is_zero(pool, c),
        Err(_) => false,
    }
}

/// `Σ_{k=lo}^{∞} term`: polynomial × geometric first, then rational series,
/// then rational × geometric (polylogarithms) and general hypergeometric
/// terms (`ₚFq`, see `hyper_series`).
pub(super) fn infinite_sum(term: ExprId, k: ExprId, lo: ExprId, pool: &ExprPool) -> Outcome {
    let term = simp(pool, term);
    match infinite_geometric(term, k, lo, pool) {
        Outcome::NotApplicable => {}
        other => return other,
    }
    match infinite_rational(term, k, lo, pool) {
        Outcome::NotApplicable => {}
        other => return other,
    }
    match super::hyper_series::infinite_polylog(term, k, lo, pool) {
        Outcome::NotApplicable => {}
        other => return other,
    }
    super::hyper_series::infinite_hypergeometric(term, k, lo, pool)
}

#[cfg(test)]
mod tests {
    use super::super::{sum_definite, SumError};
    use super::*;
    use crate::errors::AlkahestError;
    use crate::jit::eval_interp;
    use crate::kernel::expr::PredicateKind;
    use crate::simplify::{enter_assumptions, AssumptionContext};

    fn conds(s: &crate::DerivedExpr<ExprId>) -> Vec<SideCondition> {
        s.log
            .0
            .iter()
            .flat_map(|st| st.side_conditions.iter().cloned())
            .collect()
    }

    fn close(got: f64, want: f64) -> bool {
        (got - want).abs() <= 1e-10 * want.abs().max(1.0)
    }

    #[test]
    fn infinite_geometric_symbolic_ratio_is_conditional() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let r = pool.symbol("r", Domain::Real);
        let s = sum_definite(
            pool.pow(r, k),
            k,
            pool.integer(0),
            pool.pos_infinity(),
            &pool,
        )
        .expect("Σ r^k");
        let c = conds(&s);
        assert_eq!(c.len(), 1, "one |r| < 1 condition, got {c:?}");
        assert!(matches!(c[0], SideCondition::Positive(_)));
        for rv in [0.3_f64, -0.7, 0.95] {
            let mut env = HashMap::new();
            env.insert(r, rv);
            let got = eval_interp(s.value, &env, &pool).unwrap();
            assert!(close(got, 1.0 / (1.0 - rv)), "r={rv}: {got}");
        }
    }

    #[test]
    fn arithmetico_geometric_finite_and_infinite() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let n = pool.symbol("n", Domain::Real);
        let r = pool.symbol("r", Domain::Real);
        for p in 1..=3_i64 {
            let term = pool.mul(vec![pool.pow(k, pool.integer(p)), pool.pow(r, k)]);
            let fin = sum_definite(term, k, pool.integer(0), n, &pool).expect("finite");
            let inf = sum_definite(term, k, pool.integer(1), pool.pos_infinity(), &pool)
                .expect("infinite");
            for rv in [0.4_f64, -0.6, 3.0] {
                for nv in 0..7 {
                    let mut env = HashMap::new();
                    env.insert(r, rv);
                    env.insert(n, nv as f64);
                    let got = eval_interp(fin.value, &env, &pool).unwrap();
                    let want: f64 = (0..=nv)
                        .map(|j| (j as f64).powi(p as i32) * rv.powi(j))
                        .sum();
                    assert!(close(got, want), "p={p} r={rv} n={nv}: {got} vs {want}");
                }
                if rv.abs() < 1.0 {
                    let mut env = HashMap::new();
                    env.insert(r, rv);
                    let got = eval_interp(inf.value, &env, &pool).unwrap();
                    let want: f64 = (1..4000)
                        .map(|j| (j as f64).powi(p as i32) * rv.powi(j))
                        .sum();
                    assert!(close(got, want), "p={p} r={rv}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn exponential_ratio_converges_under_positive_parameters() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let ke = pool.symbol("ke", Domain::Positive);
        let tau = pool.symbol("tau", Domain::Positive);
        let term = pool.func("exp", vec![pool.mul(vec![pool.integer(-1), ke, k, tau])]);
        let s = sum_definite(term, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap();
        assert!(conds(&s).is_empty(), "ke, tau > 0 decide convergence");
        let mut env = HashMap::new();
        env.insert(ke, 0.3);
        env.insert(tau, 8.0);
        let got = eval_interp(s.value, &env, &pool).unwrap();
        assert!(close(got, 1.0 / (1.0 - (-2.4_f64).exp())));
    }

    #[test]
    fn divergent_geometric_is_refused() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let inf = pool.pos_infinity();
        let e =
            sum_definite(pool.pow(pool.integer(2), k), k, pool.integer(0), inf, &pool).unwrap_err();
        assert_eq!(e.code(), "E-SUM-003");
        // r > 1 stated: refused, not conditional.
        let r = pool.symbol("r", Domain::Real);
        let mut ctx = AssumptionContext::new();
        ctx.refine(
            pool.predicate(PredicateKind::Gt, vec![r, pool.integer(1)]),
            &pool,
        )
        .unwrap();
        let _g = enter_assumptions(&ctx, &pool);
        let e = sum_definite(pool.pow(r, k), k, pool.integer(0), inf, &pool).unwrap_err();
        assert!(matches!(e, SumError::BoundSubstitution(_)), "{e:?}");
    }

    #[test]
    fn stated_unit_interval_discharges_the_condition() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let r = pool.symbol("r", Domain::Real);
        let mut ctx = AssumptionContext::new();
        ctx.refine(
            pool.predicate(PredicateKind::Gt, vec![r, pool.integer(0)]),
            &pool,
        )
        .unwrap();
        ctx.refine(
            pool.predicate(PredicateKind::Lt, vec![r, pool.integer(1)]),
            &pool,
        )
        .unwrap();
        let _g = enter_assumptions(&ctx, &pool);
        let s = sum_definite(
            pool.pow(r, k),
            k,
            pool.integer(0),
            pool.pos_infinity(),
            &pool,
        )
        .unwrap();
        assert!(conds(&s).is_empty(), "{:?}", conds(&s));
    }

    fn rational_case(term: ExprId, lo: i64, want: f64, pool: &ExprPool, k: ExprId) {
        let s = sum_definite(term, k, pool.integer(lo), pool.pos_infinity(), pool)
            .unwrap_or_else(|e| panic!("{}: {e}", pool.display(term)));
        let got = eval_interp(s.value, &HashMap::new(), pool)
            .unwrap_or_else(|| panic!("eval {}", pool.display(s.value)));
        assert!(
            close(got, want),
            "{} → {}: {got} vs {want}",
            pool.display(term),
            pool.display(s.value)
        );
    }

    #[test]
    fn rational_series_telescope_and_zeta() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let i = |v: i64| pool.integer(v);
        let inv = |e: ExprId| pool.pow(e, i(-1));
        let pi = std::f64::consts::PI;
        // 1/(k² + k) = 1
        rational_case(inv(pool.add(vec![pool.pow(k, i(2)), k])), 1, 1.0, &pool, k);
        // 1/(4k² − 1) = 1/2
        rational_case(
            inv(pool.add(vec![pool.mul(vec![i(4), pool.pow(k, i(2))]), i(-1)])),
            1,
            0.5,
            &pool,
            k,
        );
        // (k²)^{-1}, k^{-4}, k^{-2} from 3
        rational_case(inv(pool.pow(k, i(2))), 1, pi * pi / 6.0, &pool, k);
        rational_case(pool.pow(k, i(-4)), 1, pi.powi(4) / 90.0, &pool, k);
        rational_case(pool.pow(k, i(-2)), 3, pi * pi / 6.0 - 1.25, &pool, k);
        // 1/(2k+1)² from 0 = π²/8
        rational_case(
            inv(pool.pow(pool.add(vec![pool.mul(vec![i(2), k]), i(1)]), i(2))),
            0,
            pi * pi / 8.0,
            &pool,
            k,
        );
        // 1/k³ − 1/(k+1)³ telescopes to 1
        rational_case(
            pool.add(vec![
                pool.pow(k, i(-3)),
                pool.mul(vec![i(-1), pool.pow(pool.add(vec![k, i(1)]), i(-3))]),
            ]),
            1,
            1.0,
            &pool,
            k,
        );
        // 1/(k(k+1)(k+2)) = 1/4
        rational_case(
            inv(pool.mul(vec![k, pool.add(vec![k, i(1)]), pool.add(vec![k, i(2)])])),
            1,
            0.25,
            &pool,
            k,
        );
    }

    #[test]
    fn rational_series_refusals() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let i = |v: i64| pool.integer(v);
        let inf = pool.pos_infinity();
        // Σ 1/k diverges.
        let e = sum_definite(pool.pow(k, i(-1)), k, i(1), inf, &pool).unwrap_err();
        assert_eq!(e.code(), "E-SUM-003");
        // Σ 1/k³ is named ζ(3) (G11b), Σ 1/(2k+1)³ is 7/8·ζ(3).
        let v = sum_definite(pool.pow(k, i(-3)), k, i(1), inf, &pool).unwrap();
        assert_eq!(v.value, pool.func("zeta", vec![i(3)]));
        let odd = pool.pow(pool.add(vec![pool.mul(vec![i(2), k]), i(1)]), i(-3));
        let v = sum_definite(odd, k, i(0), inf, &pool).unwrap();
        let want = pool.mul(vec![pool.rational(7, 8), pool.func("zeta", vec![i(3)])]);
        assert_eq!(
            crate::simplify::simplify(pool.add(vec![v.value, pool.mul(vec![i(-1), want])]), &pool)
                .value,
            i(0)
        );
        // A pole inside the range.
        let t = pool.pow(pool.mul(vec![k, pool.add(vec![k, i(1)])]), i(-1));
        let e = sum_definite(t, k, i(0), inf, &pool).unwrap_err();
        assert_eq!(e.code(), "E-SUM-003");
        // Irreducible quadratic denominator (F12): (1 + π·coth π)/2.
        let t = pool.pow(pool.add(vec![pool.pow(k, i(2)), i(1)]), i(-1));
        let v = sum_definite(t, k, i(0), inf, &pool).unwrap();
        let pi = std::f64::consts::PI;
        let got = eval_interp(v.value, &HashMap::new(), &pool).unwrap();
        assert!(close(got, 0.5 + pi / (2.0 * pi.tanh())), "{got}");
    }

    #[test]
    fn symbolic_shifts_carry_their_conditions() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let i = |v: i64| pool.integer(v);
        // 1/((k+a)(k+a+2)) from 0 = (1/a + 1/(a+1))/2, needs a > 0.
        let t = pool.pow(
            pool.mul(vec![pool.add(vec![k, a]), pool.add(vec![k, a, i(2)])]),
            i(-1),
        );
        let s = sum_definite(t, k, i(0), pool.pos_infinity(), &pool).unwrap();
        assert!(!conds(&s).is_empty());
        for av in [0.5_f64, 2.0, 7.25] {
            let mut env = HashMap::new();
            env.insert(a, av);
            let got = eval_interp(s.value, &env, &pool).unwrap();
            assert!(
                close(got, 0.5 * (1.0 / av + 1.0 / (av + 1.0))),
                "a={av}: {got}"
            );
        }
    }

    #[test]
    fn infinite_bounds_other_than_upper_are_refused() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let r = pool.symbol("r", Domain::Real);
        let inf = pool.pos_infinity();
        let minus_inf = pool.mul(vec![pool.integer(-1), inf]);
        assert!(sum_definite(pool.pow(r, k), k, minus_inf, pool.integer(0), &pool).is_err());
        let hi = pool.add(vec![inf, pool.integer(-1)]);
        assert!(sum_definite(pool.pow(r, k), k, pool.integer(0), hi, &pool).is_err());
    }
}
