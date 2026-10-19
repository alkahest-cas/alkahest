//! Infinite sums of hypergeometric terms beyond polynomial × geometric (F12).
//!
//! Two families, tried after `closed_form`'s geometric and rational paths:
//!
//! 1. **Rational × geometric** (the polylogarithm family),
//!    `Σ_{k≥lo} R(k)·q^k` with `R` rational in `k` and every pole of `R` an
//!    integer: `R` splits into a polynomial part (summed by the geometric
//!    antidifference) and partial fractions `c/(k − ρ)^s`, and
//!    `Σ_{k≥lo} q^k/(k − ρ)^s = q^ρ·(Li_s(q) − Σ_{j=1}^{lo−ρ−1} q^j/j^s)`.
//!    So `Σ rᵏ/k = −log(1 − r)`, `Σ rᵏ/k² = Li₂(r)`,
//!    `Σ rᵏ/(k(k+1))` is elementary. Converges for `|q| < 1` (decided from the
//!    assumptions or stated as a side condition, like the geometric series)
//!    and at `q = −1` when `R` has no polynomial part.
//!
//! 2. **General hypergeometric terms**: when `t_{k+1}/t_k` is a rational
//!    function of `k` — products of powers of polynomials whose roots are
//!    rational (or linear with symbolic shifts), `Γ`/factorial/binomial/
//!    Pochhammer of `k`-linear arguments with integer slope, and geometric
//!    factors `q^k`, `exp(c·k)` — the sum from `k₀` is `t_{k₀}·ₚFq(a; b; z)`
//!    with `ratio(k₀ + n) = z·Π(n + aᵢ)/(Π(n + bⱼ)·(n + 1))`. That node is
//!    exact; [`crate::specfun::expand_hyper`] then rewrites it into
//!    elementary or special functions where a closed form is known
//!    (`Σ qᵏ/k! = e^q`, `Σ x^{2k}/(2k)! = cosh x`,
//!    `Σ binomial(2k, k)xᵏ = (1 − 4x)^{−1/2}`), and the node is returned
//!    otherwise. Convergence: a terminating series always; `p ≤ q` for every
//!    `z`; `p = q + 1` for `|z| < 1`, at `z = 1` when `Σb − Σa > 0` (Gauss)
//!    and at `z = −1` when `Σb − Σa > −1`; `p > q + 1` diverges.

use super::closed_form::{
    affine_in_k, antidifference_at, geometric_antidifference, geometric_convergence, has_float,
    is_zero, literal, neg, partial_fractions, poly_coeffs, rational_expr, rational_piece, simp,
    tidy, Outcome, RationalPiece,
};
use super::expr_ratio::is_free_of_k;
use crate::deriv::log::SideCondition;
use crate::kernel::subs::subs;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::{assumed_sign, Sign};
use rug::{Integer, Rational};
use std::collections::HashMap;

/// Largest number of explicit `q^j/j^s` terms subtracted from a polylog.
const MAX_HEAD_TERMS: i64 = 1000;
/// Largest integer slope of a `Γ` argument (`Γ(m·k + β)` adds `m` parameters).
const MAX_GAMMA_SLOPE: i64 = 12;
/// Largest integer power of a factor.
const MAX_POW: i64 = 32;
/// Largest number of leading zero terms skipped to find `t_{k₀} ≠ 0`.
const MAX_ZERO_SKIP: i64 = 64;
/// Largest number of parameters on either side.
const MAX_PARAMS: usize = 48;

fn push_cond(conditions: &mut Vec<SideCondition>, c: SideCondition) {
    if !conditions.contains(&c) {
        conditions.push(c);
    }
}

/// Split one product into its geometric part `q` (`q^k`, the product of the
/// `base^{a}` and `exp(a)` of every `base^{a·k+b}` / `exp(a·k+b)` factor) and
/// the rest (`k`-free coefficient folded in). `None` when an exponent is not
/// affine in `k`.
fn split_geometric(piece: ExprId, k: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let factors: Vec<ExprId> = match pool.get(piece) {
        ExprData::Mul(xs) => xs.to_vec(),
        _ => vec![piece],
    };
    let mut q_parts = Vec::new();
    let mut rest = Vec::new();
    for f in factors {
        match pool.get(f) {
            ExprData::Pow { base, exp }
                if is_free_of_k(base, k, pool) && !is_free_of_k(exp, k, pool) =>
            {
                let (a, b) = affine_in_k(exp, k, pool)?;
                q_parts.push(pool.pow(base, a));
                rest.push(pool.pow(base, b));
            }
            ExprData::Func { name, args }
                if name == "exp" && args.len() == 1 && !is_free_of_k(args[0], k, pool) =>
            {
                let (a, b) = affine_in_k(args[0], k, pool)?;
                q_parts.push(pool.func("exp", vec![a]));
                rest.push(pool.func("exp", vec![b]));
            }
            _ => rest.push(f),
        }
    }
    Some((simp(pool, pool.mul(q_parts)), pool.mul(rest)))
}

// ---------------------------------------------------------------------------
// Family 1: rational × geometric, integer poles → polylogarithms
// ---------------------------------------------------------------------------

/// `num = quot·den + rem` for a monic-up-to-literal `den` (literal
/// coefficients); `None` when the leading coefficient is not a literal.
fn poly_divmod(
    num: &[ExprId],
    den: &[ExprId],
    pool: &ExprPool,
) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    let lc = literal(pool, *den.last()?)?;
    if lc == 0 {
        return None;
    }
    let inv = rational_expr(pool, &(Rational::from(1) / lc));
    let mut rem: Vec<ExprId> = num.to_vec();
    if rem.len() < den.len() {
        return Some((vec![pool.integer(0_i32)], rem));
    }
    let mut quot = vec![pool.integer(0_i32); rem.len() - den.len() + 1];
    for i in (0..quot.len()).rev() {
        let c = simp(pool, pool.mul(vec![rem[i + den.len() - 1], inv]));
        quot[i] = c;
        for (j, &d) in den.iter().enumerate() {
            rem[i + j] = simp(
                pool,
                pool.add(vec![rem[i + j], neg(pool, pool.mul(vec![c, d]))]),
            );
        }
    }
    rem.truncate(den.len() - 1);
    if rem.is_empty() {
        rem.push(pool.integer(0_i32));
    }
    Some((quot, rem))
}

/// `Σ_{k≥lo} R(k)·q^k` for rational `R` with integer poles.
pub(super) fn infinite_polylog(term: ExprId, k: ExprId, lo: ExprId, pool: &ExprPool) -> Outcome {
    let Some(lo_v) = literal(pool, lo).filter(|v| *v.denom() == 1) else {
        return Outcome::NotApplicable;
    };
    let Some(lo_i) = lo_v.numer().to_i64() else {
        return Outcome::NotApplicable;
    };
    let pieces: Vec<ExprId> = match pool.get(term) {
        ExprData::Add(xs) => xs.to_vec(),
        _ => vec![term],
    };
    // q → (polynomial part, pole → coefficients by order)
    type Group = (ExprId, Vec<ExprId>, Vec<(i64, Vec<ExprId>)>);
    let mut groups: Vec<Group> = Vec::new();
    let mut conditions = Vec::new();
    let mut any_pole = false;
    let one = pool.integer(1_i32);
    for piece in pieces {
        let Some((q, rest)) = split_geometric(piece, k, pool) else {
            return Outcome::NotApplicable;
        };
        if q == one || is_zero(pool, q) {
            return Outcome::NotApplicable;
        }
        let rest = simp(pool, rest);
        let (poly, rp) = match rational_piece(rest, k, pool) {
            Some(rp) => {
                if !rp.quads.is_empty() {
                    return Outcome::NotApplicable;
                }
                // Integer poles only.
                let mut den = vec![one];
                let mut ints = Vec::new();
                for &(rho, m) in &rp.poles {
                    let Some(v) = literal(pool, rho).filter(|v| *v.denom() == 1) else {
                        return Outcome::NotApplicable;
                    };
                    let Some(r) = v.numer().to_i64() else {
                        return Outcome::NotApplicable;
                    };
                    ints.push((r, m));
                    for _ in 0..m {
                        let Some(d) =
                            super::closed_form::poly_mul(&den, &[pool.integer(-r), one], pool)
                        else {
                            return Outcome::NotApplicable;
                        };
                        den = d;
                    }
                }
                let Some((quot, rem)) = poly_divmod(&rp.numer, &den, pool) else {
                    return Outcome::NotApplicable;
                };
                let proper = RationalPiece {
                    numer: rem,
                    poles: rp.poles.clone(),
                    quads: Vec::new(),
                };
                any_pole = true;
                (quot, Some((proper, ints)))
            }
            None => match poly_coeffs(rest, k, pool) {
                Some(p) => (p, None),
                None => return Outcome::NotApplicable,
            },
        };
        let slot = match groups.iter().position(|g| g.0 == q) {
            Some(i) => i,
            None => {
                groups.push((q, Vec::new(), Vec::new()));
                groups.len() - 1
            }
        };
        let g = &mut groups[slot];
        if poly.len() > g.1.len() {
            g.1.resize(poly.len(), pool.integer(0_i32));
        }
        for (i, c) in poly.into_iter().enumerate() {
            g.1[i] = simp(pool, pool.add(vec![g.1[i], c]));
        }
        if let Some((proper, ints)) = rp {
            let Some(pf) = partial_fractions(&proper, k, pool, &mut conditions) else {
                return Outcome::NotApplicable;
            };
            for ((_, cs), (r, _)) in pf.into_iter().zip(ints) {
                match g.2.iter_mut().find(|e| e.0 == r) {
                    Some(e) => {
                        if cs.len() > e.1.len() {
                            e.1.resize(cs.len(), pool.integer(0_i32));
                        }
                        for (s, c) in cs.into_iter().enumerate() {
                            e.1[s] = simp(pool, pool.add(vec![e.1[s], c]));
                        }
                    }
                    None => g.2.push((r, cs)),
                }
            }
        }
    }
    if !any_pole {
        // Polynomial × geometric: `closed_form` already had its say.
        return Outcome::NotApplicable;
    }
    let mut parts = Vec::new();
    for (q, poly, poles) in &groups {
        let has_poly = poly.iter().any(|&c| !is_zero(pool, c));
        // Convergence.
        match literal(pool, *q) {
            Some(v) if v == -1 => {
                if has_poly {
                    return Outcome::Refused(
                        "the series diverges: its terms do not tend to 0 \
                         (a polynomial times (-1)^k)"
                            .to_string(),
                    );
                }
            }
            _ => match geometric_convergence(*q, pool) {
                Err(why) => return Outcome::Refused(why),
                Ok(Some(c)) => push_cond(&mut conditions, c),
                Ok(None) => {}
            },
        }
        if has_poly {
            let a = geometric_antidifference(poly, *q, pool);
            parts.push(neg(pool, antidifference_at(&a, *q, lo, pool)));
        }
        for (rho, cs) in poles {
            let j0 = lo_i - rho;
            if j0 <= 0 {
                if cs.iter().all(|&c| is_zero(pool, c)) {
                    continue;
                }
                return Outcome::Refused(format!(
                    "the summand is undefined at k = {rho}, which lies inside the summation range"
                ));
            }
            if j0 - 1 > MAX_HEAD_TERMS {
                return Outcome::NotApplicable;
            }
            let shift = pool.pow(*q, pool.integer(*rho));
            for (s_idx, &c) in cs.iter().enumerate() {
                if is_zero(pool, c) {
                    continue;
                }
                let s = s_idx as i64 + 1;
                let Ok(li) = crate::specfun::polylog(pool.integer(s), *q, pool) else {
                    return Outcome::NotApplicable;
                };
                let mut head = Vec::new();
                for j in 1..j0 {
                    if crate::budget::check().is_err() {
                        return Outcome::NotApplicable;
                    }
                    head.push(neg(
                        pool,
                        pool.mul(vec![
                            pool.pow(*q, pool.integer(j)),
                            pool.rational(
                                Integer::from(1),
                                Integer::from(Integer::u_pow_u(j as u32, s as u32)),
                            ),
                        ]),
                    ));
                }
                head.push(li);
                parts.push(pool.mul(vec![c, shift, pool.add(head)]));
            }
        }
    }
    let value = tidy(simp(pool, pool.add(parts)), pool);
    if has_float(value, pool) && !has_float(term, pool) {
        return Outcome::NotApplicable;
    }
    Outcome::Sum {
        value,
        conditions,
        rule: "polylog_series",
    }
}

// ---------------------------------------------------------------------------
// Family 2: general hypergeometric terms → pFq
// ---------------------------------------------------------------------------

/// `t_{k+1}/t_k = Π scale · Π(k + upᵢ) / Π(k + downⱼ)`, every entry free of `k`.
#[derive(Default)]
struct Ratio {
    scale: Vec<ExprId>,
    ups: Vec<ExprId>,
    downs: Vec<ExprId>,
}

fn ratio_of(e: ExprId, k: ExprId, pool: &ExprPool, depth: u32) -> Option<Ratio> {
    if depth > 48 {
        return None;
    }
    if is_free_of_k(e, k, pool) {
        return Some(Ratio::default());
    }
    if e == k {
        return Some(Ratio {
            scale: Vec::new(),
            ups: vec![pool.integer(1_i32)],
            downs: vec![pool.integer(0_i32)],
        });
    }
    match pool.get(e) {
        ExprData::Mul(xs) => {
            let mut acc = Ratio::default();
            for &x in xs.iter() {
                let r = ratio_of(x, k, pool, depth + 1)?;
                acc.scale.extend(r.scale);
                acc.ups.extend(r.ups);
                acc.downs.extend(r.downs);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            if is_free_of_k(base, k, pool) {
                let (a, _) = affine_in_k(exp, k, pool)?;
                return Some(Ratio {
                    scale: vec![pool.pow(base, a)],
                    ..Ratio::default()
                });
            }
            let n = match pool.get(exp) {
                ExprData::Integer(n) => n.0.to_i64()?,
                _ => return None,
            };
            if n == 0 || n.abs() > MAX_POW {
                return None;
            }
            let r = ratio_of(base, k, pool, depth + 1)?;
            let mut out = Ratio::default();
            let (ups, downs) = if n > 0 {
                (&r.ups, &r.downs)
            } else {
                (&r.downs, &r.ups)
            };
            for _ in 0..n.abs() {
                out.ups.extend(ups.iter().copied());
                out.downs.extend(downs.iter().copied());
            }
            for s in r.scale {
                out.scale.push(pool.pow(s, pool.integer(n)));
            }
            Some(out)
        }
        ExprData::Func { name, args } => {
            let one = pool.integer(1_i32);
            match (name.as_str(), args.len()) {
                ("exp", 1) => {
                    let (a, _) = affine_in_k(args[0], k, pool)?;
                    Some(Ratio {
                        scale: vec![pool.func("exp", vec![a])],
                        ..Ratio::default()
                    })
                }
                ("gamma", 1) => gamma_ratio(args[0], k, pool),
                ("factorial", 1) => gamma_ratio(pool.add(vec![args[0], one]), k, pool),
                ("binomial", 2) => {
                    let g = |x: ExprId| pool.func("gamma", vec![x]);
                    let inv = |x: ExprId| pool.pow(x, pool.integer(-1_i32));
                    let rest = pool.add(vec![args[0], neg(pool, args[1]), one]);
                    let e2 = pool.mul(vec![
                        g(pool.add(vec![args[0], one])),
                        inv(g(pool.add(vec![args[1], one]))),
                        inv(g(rest)),
                    ]);
                    ratio_of(e2, k, pool, depth + 1)
                }
                ("pochhammer", 2) => {
                    let e2 = pool.mul(vec![
                        pool.func("gamma", vec![pool.add(vec![args[0], args[1]])]),
                        pool.pow(pool.func("gamma", vec![args[0]]), pool.integer(-1_i32)),
                    ]);
                    ratio_of(e2, k, pool, depth + 1)
                }
                _ => None,
            }
        }
        // A polynomial in k (an `Add`).
        _ => poly_ratio(e, k, pool),
    }
}

/// `L(k+1)/L(k)` for a polynomial `L`: linear with any literal slope and a
/// shift free of `k`, or literal coefficients splitting into rational linear
/// factors.
fn poly_ratio(e: ExprId, k: ExprId, pool: &ExprPool) -> Option<Ratio> {
    let c = poly_coeffs(e, k, pool)?;
    if c.len() < 2 {
        return None;
    }
    let one = pool.integer(1_i32);
    let mut out = Ratio::default();
    if c.len() == 2 {
        // A·k + B = A·(k + B/A).
        let a = literal(pool, c[1]).filter(|a| *a != 0)?;
        let shift = simp(
            pool,
            pool.mul(vec![c[0], rational_expr(pool, &(Rational::from(1) / a))]),
        );
        out.ups.push(simp(pool, pool.add(vec![shift, one])));
        out.downs.push(shift);
        return Some(out);
    }
    let lits: Vec<Rational> = c.iter().map(|&x| literal(pool, x)).collect::<Option<_>>()?;
    let pe = {
        let p = ExprPool::new();
        let x = p.symbol("x", crate::kernel::Domain::Real);
        let ints = {
            let mut den = Integer::from(1);
            for c in &lits {
                den = den.lcm(c.denom());
            }
            lits.iter()
                .map(|c| (c.clone() * Rational::from(den.clone())).numer().clone())
                .collect::<Vec<_>>()
        };
        let terms: Vec<ExprId> = ints
            .iter()
            .enumerate()
            .map(|(i, c)| p.mul(vec![p.integer(c.clone()), p.pow(x, p.integer(i as i64))]))
            .collect();
        let up = crate::poly::UniPoly::from_symbolic(p.add(terms), x, &p).ok()?;
        up.factor_z().ok()?
    };
    for (f, m) in &pe.factors {
        if f.degree() != 1 {
            return None;
        }
        let cf = f.coefficients();
        // root ρ = −c₀/c₁: factor (k − ρ) = (k + c₀/c₁).
        let shift = Rational::from((cf[0].clone(), cf[1].clone()));
        for _ in 0..*m {
            out.ups
                .push(rational_expr(pool, &(shift.clone() + Rational::from(1))));
            out.downs.push(rational_expr(pool, &shift));
        }
    }
    Some(out)
}

/// `Γ(α(k+1) + β)/Γ(αk + β)` for an integer slope `α ≠ 0`.
fn gamma_ratio(arg: ExprId, k: ExprId, pool: &ExprPool) -> Option<Ratio> {
    let (a, b) = affine_in_k(arg, k, pool)?;
    let a = simp(pool, a);
    let b = simp(pool, b);
    let alpha = literal(pool, a)
        .filter(|v| *v.denom() == 1)?
        .numer()
        .to_i64()?;
    if alpha == 0 || alpha.abs() > MAX_GAMMA_SLOPE {
        return None;
    }
    let mut out = Ratio::default();
    if alpha > 0 {
        // Π_{t<α} (αk + β + t) = α^α Π (k + (β + t)/α)
        out.scale
            .push(pool.pow(pool.integer(alpha), pool.integer(alpha)));
        for t in 0..alpha {
            out.ups.push(simp(
                pool,
                pool.mul(vec![
                    pool.add(vec![b, pool.integer(t)]),
                    pool.rational(1, alpha),
                ]),
            ));
        }
    } else {
        // m = −α: Γ(β − mk − m)/Γ(β − mk) = 1/Π_{t=1}^{m} (β − t − mk)
        //       = 1/((−m)^m Π (k − (β − t)/m)).
        let m = -alpha;
        out.scale.push(pool.pow(pool.integer(-m), pool.integer(-m)));
        for t in 1..=m {
            out.downs.push(simp(
                pool,
                pool.mul(vec![
                    pool.add(vec![b, pool.integer(-t)]),
                    pool.rational(-1, m),
                ]),
            ));
        }
    }
    Some(out)
}

/// The `N ≥ 0` with `e = −N` for a literal non-positive integer `e`.
fn nonpositive_int(pool: &ExprPool, e: ExprId) -> Option<i64> {
    let v = literal(pool, e)?;
    if *v.denom() == 1 && v <= 0 {
        (-v.numer().clone()).to_i64()
    } else {
        None
    }
}

/// `t` is not a value: a division by zero, `Γ` at a pole, or an infinity.
fn undefined_value(e: ExprId, pool: &ExprPool) -> bool {
    if e == pool.pos_infinity() {
        return true;
    }
    match pool.get(e) {
        ExprData::Pow { base, exp } => {
            let zero_base = is_zero(pool, base);
            let negative = literal(pool, exp).is_some_and(|v| v < 0);
            (zero_base && negative) || undefined_value(base, pool) || undefined_value(exp, pool)
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&x| undefined_value(x, pool)),
        ExprData::Func { name, args } => {
            let pole = matches!(name.as_str(), "gamma" | "digamma" | "trigamma")
                && args.len() == 1
                && nonpositive_int(pool, args[0]).is_some();
            let fact_pole = name == "factorial"
                && args.len() == 1
                && literal(pool, args[0]).is_some_and(|v| *v.denom() == 1 && v < 0);
            pole || fact_pole || args.iter().any(|&x| undefined_value(x, pool))
        }
        ExprData::Symbol { name, .. } => matches!(name.as_str(), "zoo" | "nan" | "oo"),
        _ => false,
    }
}

fn at(term: ExprId, k: ExprId, v: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(k, v);
    let e = simp(pool, subs(term, &m, pool));
    simp(pool, fold_combinatorial(e, pool, 0))
}

fn small_nonneg(pool: &ExprPool, e: ExprId) -> Option<u32> {
    let v = literal(pool, e)?;
    if *v.denom() != 1 || v < 0 {
        return None;
    }
    v.numer().to_u32().filter(|&n| n <= 2000)
}

/// `factorial(n)`, `binomial(n, m)` and `pochhammer(a, n)` at literal
/// arguments, as numbers (`simplify` keeps them as nodes).
fn fold_combinatorial(e: ExprId, pool: &ExprPool, depth: u32) -> ExprId {
    if depth > 64 {
        return e;
    }
    match pool.get(e) {
        ExprData::Add(xs) => pool.add(
            xs.iter()
                .map(|&x| fold_combinatorial(x, pool, depth + 1))
                .collect(),
        ),
        ExprData::Mul(xs) => pool.mul(
            xs.iter()
                .map(|&x| fold_combinatorial(x, pool, depth + 1))
                .collect(),
        ),
        ExprData::Pow { base, exp } => pool.pow(fold_combinatorial(base, pool, depth + 1), exp),
        ExprData::Func { name, args } => {
            match (name.as_str(), args.len()) {
                ("factorial", 1) => {
                    if let Some(n) = small_nonneg(pool, args[0]) {
                        return pool.integer(Integer::from(Integer::factorial(n)));
                    }
                }
                ("binomial", 2) => {
                    if let (Some(n), Some(m)) =
                        (small_nonneg(pool, args[0]), small_nonneg(pool, args[1]))
                    {
                        return pool.integer(Integer::from(n).binomial(m));
                    }
                }
                ("pochhammer", 2) => {
                    if let (Some(a), Some(n)) =
                        (literal(pool, args[0]), small_nonneg(pool, args[1]))
                    {
                        let mut acc = Rational::from(1);
                        for j in 0..n {
                            acc *= a.clone() + Rational::from(j);
                        }
                        return rational_expr(pool, &acc);
                    }
                }
                _ => {}
            }
            e
        }
        _ => e,
    }
}

/// A numeric value of a variable-free expression, when it has one.
fn number(e: ExprId, pool: &ExprPool) -> Option<f64> {
    let v = crate::jit::eval_interp(e, &HashMap::new(), pool)?;
    v.is_finite().then_some(v)
}

/// `Σ_{k≥lo} t_k` as `t_{k₀}·ₚFq(a; b; z)`, expanded when a closed form is known.
pub(super) fn infinite_hypergeometric(
    term: ExprId,
    k: ExprId,
    lo: ExprId,
    pool: &ExprPool,
) -> Outcome {
    let Some(ratio) = ratio_of(term, k, pool, 0) else {
        return Outcome::NotApplicable;
    };
    if ratio.ups.len() > MAX_PARAMS || ratio.downs.len() > MAX_PARAMS {
        return Outcome::NotApplicable;
    }
    // First non-zero term: leading factors like `1/Γ(k − 2)` or `k` vanish.
    let mut k0 = lo;
    let mut t0 = at(term, k, k0, pool);
    let mut skipped = 0;
    loop {
        if undefined_value(t0, pool) {
            return Outcome::Refused(format!(
                "the summand is undefined at k = {}, which lies inside the summation range",
                pool.display(k0)
            ));
        }
        if !is_zero(pool, t0) {
            break;
        }
        skipped += 1;
        if skipped > MAX_ZERO_SKIP {
            return Outcome::NotApplicable;
        }
        k0 = simp(pool, pool.add(vec![k0, pool.integer(1_i32)]));
        t0 = at(term, k, k0, pool);
    }
    let shift = |x: ExprId| simp(pool, pool.add(vec![k0, x]));
    let mut a: Vec<ExprId> = ratio.ups.iter().map(|&x| shift(x)).collect();
    a.push(pool.integer(1_i32));
    let mut b: Vec<ExprId> = ratio.downs.iter().map(|&x| shift(x)).collect();
    let z = simp(pool, pool.mul(ratio.scale.clone()));
    if is_zero(pool, z) {
        return Outcome::Sum {
            value: t0,
            conditions: Vec::new(),
            rule: "hypergeometric_series",
        };
    }
    let terminates = a.iter().any(|&x| nonpositive_int(pool, x).is_some());
    let lower_pole = b.iter().copied().find_map(|x| nonpositive_int(pool, x));
    if let Some(m) = lower_pole {
        // The ratio divides by zero at n = m: look at the term itself there.
        if terminates {
            // A zero and a pole of the ratio: the recurrence does not determine
            // the later terms (`(k − 2)·qᵏ` restarts after its zero).
            return Outcome::NotApplicable;
        }
        let kbad = simp(pool, pool.add(vec![k0, pool.integer(m + 1)]));
        if undefined_value(at(term, k, kbad, pool), pool) {
            return Outcome::Refused(format!(
                "the summand is undefined at k = {}, which lies inside the summation range",
                pool.display(kbad)
            ));
        }
        return Outcome::NotApplicable;
    }
    // Cancel common parameters (no non-positive integers are on both sides).
    let mut i = 0;
    while i < a.len() {
        if let Some(j) = b.iter().position(|&y| y == a[i]) {
            a.remove(i);
            b.remove(j);
        } else {
            i += 1;
        }
    }
    let mut conditions: Vec<SideCondition> = Vec::new();
    // A symbolic lower parameter must not be a non-positive integer.
    for &y in &b {
        if literal(pool, y).is_none() && assumed_sign(y, pool) != Some(Sign::Positive) {
            push_cond(&mut conditions, SideCondition::Positive(y));
        }
    }
    let (p, q) = (a.len(), b.len());
    if !terminates {
        if p > q + 1 {
            return Outcome::Refused(format!(
                "the series diverges: the ratio of consecutive terms grows like k^{} \
                 (a {p}F{q} series with a nonzero argument)",
                p - q - 1
            ));
        }
        if p == q + 1 {
            let excess = simp(
                pool,
                pool.add(
                    b.iter()
                        .copied()
                        .chain(a.iter().map(|&x| neg(pool, x)))
                        .collect::<Vec<_>>(),
                ),
            );
            let zv = literal(pool, z);
            let boundary = zv.as_ref().filter(|v| **v == 1 || **v == -1).cloned();
            if let Some(zv) = boundary {
                // Gauss (z = 1): Σb − Σa > 0; alternating (z = −1): > −1.
                let gap = if zv == 1 {
                    excess
                } else {
                    simp(pool, pool.add(vec![excess, pool.integer(1_i32)]))
                };
                match literal(pool, gap) {
                    Some(g) if g > 0 => {}
                    Some(_) => {
                        return Outcome::Refused(format!(
                            "the series diverges: at z = {} its terms decay like k^({}), \
                             too slowly to converge",
                            zv,
                            pool.display(simp(pool, neg(pool, excess)))
                        ))
                    }
                    None => push_cond(&mut conditions, SideCondition::Positive(gap)),
                }
            } else {
                match geometric_convergence(z, pool) {
                    Err(why) => return Outcome::Refused(why),
                    Ok(Some(c)) => push_cond(&mut conditions, c),
                    Ok(None) => {}
                }
            }
        }
    }
    let node = match crate::specfun::hyper(&a, &b, z, pool) {
        Ok(n) => n,
        Err(e) => return Outcome::Refused(e.to_string()),
    };
    let exact = simp(pool, pool.mul(vec![t0, node]));
    let mut value = exact;
    let mut rule = "hypergeometric_series";
    if let Ok(expanded) = crate::specfun::expand_hyper(exact, pool) {
        let expanded = simp(pool, expanded);
        if expanded != exact && !has_float(expanded, pool) {
            // When the result is a number, the closed form must agree with
            // the series (Arb) — a branch slip in a closed form would
            // otherwise go unnoticed.
            let accept = if has_parameter(expanded, pool) {
                true
            } else {
                match (number(expanded, pool), number(exact, pool)) {
                    (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * y.abs().max(1.0),
                    _ => false,
                }
            };
            if accept {
                value = expanded;
                rule = "hypergeometric_series_closed_form";
            }
        }
    }
    if has_float(value, pool) && !has_float(term, pool) {
        return Outcome::NotApplicable;
    }
    Outcome::Sum {
        value,
        conditions,
        rule,
    }
}

/// Mentions a symbol that is not a named constant (`pi`, `EulerGamma`, …).
fn has_parameter(e: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Symbol { name, .. } => {
            crate::eval::symbols::constant_value_f64(name.as_str()).is_none()
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&x| has_parameter(x, pool)),
        ExprData::Pow { base, exp } => has_parameter(base, pool) || has_parameter(exp, pool),
        ExprData::Func { args, .. } => args.iter().any(|&x| has_parameter(x, pool)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::sum_definite;
    use super::*;
    use crate::errors::AlkahestError;
    use crate::jit::eval_interp;
    use crate::kernel::Domain;

    fn conds(s: &crate::DerivedExpr<ExprId>) -> Vec<SideCondition> {
        s.log
            .0
            .iter()
            .flat_map(|st| st.side_conditions.iter().cloned())
            .collect()
    }

    fn close(got: f64, want: f64, tol: f64) -> bool {
        (got - want).abs() <= tol * want.abs().max(1.0)
    }

    /// Direct partial sum `Σ_{k=lo}^{lo+n} f(k)`.
    fn partial(f: impl Fn(f64) -> f64, lo: i64, n: i64) -> f64 {
        (lo..lo + n).map(|k| f(k as f64)).sum()
    }

    fn eval_at(e: ExprId, env: &[(ExprId, f64)], pool: &ExprPool) -> f64 {
        let m: HashMap<ExprId, f64> = env.iter().copied().collect();
        eval_interp(e, &m, pool).expect("numeric value")
    }

    #[test]
    fn exponential_series() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let q = pool.symbol("q", Domain::Real);
        let t = pool.mul(vec![
            pool.pow(q, k),
            pool.pow(pool.func("factorial", vec![k]), pool.integer(-1)),
        ]);
        let s = sum_definite(t, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap();
        assert_eq!(
            s.value,
            pool.func("exp", vec![q]),
            "{}",
            pool.display(s.value)
        );
        assert!(conds(&s).is_empty());
    }

    #[test]
    fn log_and_polylog_series() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let r = pool.symbol("r", Domain::Real);
        for s_ord in 1..=4_i64 {
            let t = pool.mul(vec![pool.pow(r, k), pool.pow(k, pool.integer(-s_ord))]);
            let s = sum_definite(t, k, pool.integer(1), pool.pos_infinity(), &pool).unwrap();
            assert_eq!(conds(&s).len(), 1, "|r| < 1 is stated");
            for rv in [0.3_f64, -0.6, 0.9] {
                let got = eval_at(s.value, &[(r, rv)], &pool);
                let want = partial(|kk| rv.powf(kk) / kk.powi(s_ord as i32), 1, 4000);
                assert!(close(got, want, 1e-9), "s={s_ord} r={rv}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn rational_times_geometric() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let r = pool.symbol("r", Domain::Real);
        // r^k / (k (k+1)), k ≥ 1, and k·r^k/(k+2)^2, k ≥ 0
        let t1 = pool.mul(vec![
            pool.pow(r, k),
            pool.pow(k, pool.integer(-1)),
            pool.pow(pool.add(vec![k, pool.integer(1)]), pool.integer(-1)),
        ]);
        let t2 = pool.mul(vec![
            k,
            pool.pow(r, k),
            pool.pow(pool.add(vec![k, pool.integer(2)]), pool.integer(-2)),
        ]);
        let s1 = sum_definite(t1, k, pool.integer(1), pool.pos_infinity(), &pool).unwrap();
        let s2 = sum_definite(t2, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap();
        for rv in [0.25_f64, -0.5, 0.8] {
            let w1 = partial(|kk| rv.powf(kk) / (kk * (kk + 1.0)), 1, 4000);
            let w2 = partial(|kk| kk * rv.powf(kk) / (kk + 2.0).powi(2), 0, 4000);
            assert!(close(eval_at(s1.value, &[(r, rv)], &pool), w1, 1e-9));
            assert!(close(eval_at(s2.value, &[(r, rv)], &pool), w2, 1e-9));
        }
    }

    #[test]
    fn alternating_and_divergent_boundaries() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let m1 = pool.integer(-1);
        // Σ (−1)^k / k = −log 2
        let t = pool.mul(vec![pool.pow(m1, k), pool.pow(k, pool.integer(-1))]);
        let s = sum_definite(t, k, pool.integer(1), pool.pos_infinity(), &pool).unwrap();
        assert!(close(eval_at(s.value, &[], &pool), -(2f64.ln()), 1e-12));
        // Σ 2^k / k diverges; Σ (−1)^k·k/(k+1) diverges.
        let t = pool.mul(vec![
            pool.pow(pool.integer(2), k),
            pool.pow(k, pool.integer(-1)),
        ]);
        let e = sum_definite(t, k, pool.integer(1), pool.pos_infinity(), &pool).unwrap_err();
        assert_eq!(e.code(), "E-SUM-003");
        let t = pool.mul(vec![
            pool.pow(m1, k),
            k,
            pool.pow(pool.add(vec![k, pool.integer(1)]), pool.integer(-1)),
        ]);
        let e = sum_definite(t, k, pool.integer(1), pool.pos_infinity(), &pool).unwrap_err();
        assert_eq!(e.code(), "E-SUM-003");
        // A pole inside the range.
        let t = pool.mul(vec![
            pool.pow(pool.rational(1, 2), k),
            pool.pow(pool.add(vec![k, pool.integer(-3)]), pool.integer(-1)),
        ]);
        let e = sum_definite(t, k, pool.integer(1), pool.pos_infinity(), &pool).unwrap_err();
        assert_eq!(e.code(), "E-SUM-003");
    }

    #[test]
    fn hypergeometric_closed_forms() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let x = pool.symbol("x", Domain::Real);
        let fact = |e: ExprId| pool.func("factorial", vec![e]);
        let inv = |e: ExprId| pool.pow(e, pool.integer(-1));
        let two_k = pool.mul(vec![pool.integer(2), k]);
        // Σ x^{2k}/(2k)! = cosh x
        let t = pool.mul(vec![pool.pow(x, two_k), inv(fact(two_k))]);
        let s = sum_definite(t, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap();
        for xv in [0.3_f64, 1.7, -2.2] {
            assert!(close(eval_at(s.value, &[(x, xv)], &pool), xv.cosh(), 1e-12));
        }
        assert!(conds(&s).is_empty());
        // Σ (−1)^k/(2k+1) = π/4 (Leibniz)
        let t = pool.mul(vec![
            pool.pow(pool.integer(-1), k),
            inv(pool.add(vec![two_k, pool.integer(1)])),
        ]);
        let s = sum_definite(t, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap();
        assert!(close(
            eval_at(s.value, &[], &pool),
            std::f64::consts::FRAC_PI_4,
            1e-12
        ));
        // Σ binomial(2k, k) x^k = 1/sqrt(1 − 4x), |4x| < 1 stated.
        let t = pool.mul(vec![pool.func("binomial", vec![two_k, k]), pool.pow(x, k)]);
        let s = sum_definite(t, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap();
        assert_eq!(conds(&s).len(), 1);
        for xv in [0.1_f64, -0.2] {
            let want = 1.0 / (1.0 - 4.0 * xv).sqrt();
            assert!(close(eval_at(s.value, &[(x, xv)], &pool), want, 1e-12));
        }
        // Σ k/k! x^k = x e^x (leading zero term skipped)
        let t = pool.mul(vec![k, pool.pow(x, k), inv(fact(k))]);
        let s = sum_definite(t, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap();
        for xv in [0.5_f64, -1.5] {
            assert!(close(
                eval_at(s.value, &[(x, xv)], &pool),
                xv * xv.exp(),
                1e-12
            ));
        }
    }

    #[test]
    fn hypergeometric_node_when_no_closed_form() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let fact = |e: ExprId| pool.func("factorial", vec![e]);
        let inv = |e: ExprId| pool.pow(e, pool.integer(-1));
        // Σ 1/(k!)^2 = I_0(2) — or the 0F1 node; either way the right number.
        let t = inv(pool.pow(fact(k), pool.integer(2)));
        let s = sum_definite(t, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap();
        let want = partial(
            |kk| 1.0 / (1..=kk as i64).map(|j| j as f64).product::<f64>().powi(2),
            0,
            40,
        );
        assert!(close(eval_at(s.value, &[], &pool), want, 1e-12));
        // Σ (k!)^3/(3k)!: a 3F2 at z = 1/27 with no closed form in the table.
        let t = pool.mul(vec![
            pool.pow(fact(k), pool.integer(3)),
            inv(fact(pool.mul(vec![pool.integer(3), k]))),
        ]);
        let s = sum_definite(t, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap();
        let mut want = 0.0;
        let mut term = 1.0_f64;
        for kk in 0..60 {
            want += term;
            let kf = kk as f64;
            term *= (kf + 1.0).powi(3) / ((3.0 * kf + 1.0) * (3.0 * kf + 2.0) * (3.0 * kf + 3.0));
        }
        assert!(close(eval_at(s.value, &[], &pool), want, 1e-12));
        // Σ k!·x^k diverges (2F0).
        let x = pool.symbol("x", Domain::Real);
        let t = pool.mul(vec![fact(k), pool.pow(x, k)]);
        let e = sum_definite(t, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap_err();
        assert_eq!(e.code(), "E-SUM-003");
    }

    #[test]
    fn quadratic_denominators() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let inv = |e: ExprId| pool.pow(e, pool.integer(-1));
        let k2 = pool.pow(k, pool.integer(2));
        // Σ_{k≥1} 1/(k² + a²) = (π a coth(π a) − 1)/(2a²), condition a² > 0.
        let t = inv(pool.add(vec![k2, pool.pow(a, pool.integer(2))]));
        let s = sum_definite(t, k, pool.integer(1), pool.pos_infinity(), &pool).unwrap();
        assert_eq!(conds(&s).len(), 1);
        for av in [0.5_f64, 1.0, 2.5, -1.3] {
            let pa = std::f64::consts::PI * av;
            let want = (pa / pa.tanh() - 1.0) / (2.0 * av * av);
            assert!(close(eval_at(s.value, &[(a, av)], &pool), want, 1e-12));
        }
        // Σ_{k≥0} 1/(k² + 1)
        let t = inv(pool.add(vec![k2, pool.integer(1)]));
        let s = sum_definite(t, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap();
        let pi = std::f64::consts::PI;
        assert!(close(
            eval_at(s.value, &[], &pool),
            0.5 + pi / (2.0 * pi.tanh()),
            1e-12
        ));
        // Σ_{k≥0} 1/(k² + k + 1) — α = −1/2, half-integer shift: tanh.
        let t = inv(pool.add(vec![k2, k, pool.integer(1)]));
        let s = sum_definite(t, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap();
        let want = partial(|kk| 1.0 / (kk * kk + kk + 1.0), 0, 2_000_000) + 1.0 / 2_000_000.0;
        assert!(close(eval_at(s.value, &[], &pool), want, 1e-9));
        // Σ_{k≥1} k/(k⁴ + 4) = 3/8 (Sophie Germain: two quadratics).
        let t = pool.mul(vec![
            k,
            inv(pool.add(vec![pool.pow(k, pool.integer(4)), pool.integer(4)])),
        ]);
        let s = sum_definite(t, k, pool.integer(1), pool.pos_infinity(), &pool).unwrap();
        assert!(close(eval_at(s.value, &[], &pool), 3.0 / 8.0, 1e-12));
        // Σ k/(k² + 1) diverges (the 1/k tail).
        let t = pool.mul(vec![k, inv(pool.add(vec![k2, pool.integer(1)]))]);
        let e = sum_definite(t, k, pool.integer(1), pool.pos_infinity(), &pool).unwrap_err();
        assert_eq!(e.code(), "E-SUM-003");
        // Σ 1/((k+1)(k²+1)) needs Re ψ(1 + i): refused, not guessed.
        let t = pool.mul(vec![
            inv(pool.add(vec![k, pool.integer(1)])),
            inv(pool.add(vec![k2, pool.integer(1)])),
        ]);
        let e = sum_definite(t, k, pool.integer(0), pool.pos_infinity(), &pool).unwrap_err();
        assert_eq!(e.code(), "E-SUM-002");
    }
}
