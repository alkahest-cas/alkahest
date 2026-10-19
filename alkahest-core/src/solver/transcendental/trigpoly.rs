//! Equations polynomial in `sin`, `cos` and `tan` of integer multiples of one
//! angle.
//!
//! `2·sin²x = 1`, `sin x = cos x`, `sin 2x = cos x`, `sin x·cos x = 1/4`,
//! `tan²x = 3`, `√3·sin x + cos x = 1`: every kernel argument is `Nᵢ·θ + γᵢ`
//! for one angle `θ = (a/D)·x + β` and small integers `Nᵢ`, so the equation is
//! a rational function of `S = sin θ` and `C = cos θ` (multiple angles by de
//! Moivre, `tan = sin/cos`). Its numerator `P(S, C)` is factored over ℚ and
//! each factor is solved on the circle `S² + C² = 1` by the first route that
//! applies:
//!
//! 1. reduced modulo the circle it involves `S` alone: `sin θ = r`, giving
//!    `asin r + 2πk` and `π − asin r + 2πk`;
//! 2. it involves `C` alone: `±acos r + 2πk`;
//! 3. all its terms have degrees of one parity: homogenised with `S² + C²` and
//!    divided by `Cᵈ` it is a polynomial in `T = tan θ`, giving `atan r + πk`,
//!    and `π/2 + πk` when `C = 0` is a root;
//! 4. otherwise the half-angle substitution `t = tan(θ/2)` gives
//!    `2·atan r + 2πk`, and `θ = π + 2πk` — the one point the substitution
//!    cannot reach — is checked directly.
//!
//! The univariate polynomials go to the polynomial solver (radicals, or
//! `RootOf`). Over the reals only real roots in range are kept; in the complex
//! mode every root is kept except the values `tan θ` and `tan(θ/2)` never take
//! (`±i`). Families are de-duplicated and merged (`π/4 + 2πk` and
//! `5π/4 + 2πk` are `π/4 + πk`), a family at which a `tan` of the equation is
//! undefined is dropped, and every family is substituted back into the
//! equation as written before it is returned.

use super::*;
use crate::ball::IntervalEval;
use crate::deriv::log::SideCondition;
use crate::kernel::Domain;
use crate::poly::groebner::GbPoly;
use crate::solver::verify::CBallEval;
use crate::solver::SolutionSet;
use rug::Integer;
use std::collections::HashMap;

/// Largest `|N|` accepted in `sin(N·θ)` (the degree of its expansion).
const MAX_MULTIPLE: i64 = 12;

/// Largest total degree in `S, C` of the cleared numerator.
const MAX_DEGREE: u32 = 24;

/// Precision of the rigorous range tests.
const BALL_PREC: u32 = 256;

/// The period of a solution family.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Period {
    Pi,
    TwoPi,
}

/// `θ = offset + period·k`, `k ∈ ℤ`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Family {
    pub(super) offset: ExprId,
    pub(super) period: Period,
}

impl Period {
    fn expr(self, pool: &ExprPool) -> ExprId {
        let pi = crate::eval::symbols::pi_symbol(pool);
        match self {
            Period::Pi => pi,
            Period::TwoPi => pool.mul(vec![pool.integer(2_i32), pi]),
        }
    }

    fn value(self) -> f64 {
        match self {
            Period::Pi => std::f64::consts::PI,
            Period::TwoPi => 2.0 * std::f64::consts::PI,
        }
    }
}

/// What the routes share: the indeterminates `[S, C, params…]` of the cleared
/// numerator, the map back from frozen atoms to what they stood for, and the
/// domain.
struct Ctx<'a> {
    pool: &'a ExprPool,
    vars: Vec<ExprId>,
    thaw: HashMap<ExprId, ExprId>,
    complex: bool,
}

impl Ctx<'_> {
    fn thaw(&self, e: ExprId) -> ExprId {
        if self.thaw.is_empty() {
            return e;
        }
        crate::kernel::subs::subs(e, &self.thaw, self.pool)
    }

    /// `p`, free of `S` and `C`, as an expression in the original atoms.
    fn param_expr(&self, p: &GbPoly) -> Option<ExprId> {
        let e = crate::solver::gbpoly_to_expr(p, &self.vars, self.pool)?;
        Some(simp(self.thaw(e), self.pool))
    }

    /// Record `p ≠ 0` for a polynomial in the parameters that is not a
    /// constant: the generic reading of a coefficient that a special parameter
    /// value would kill.
    fn assume_param_nonzero(&self, p: &GbPoly) {
        if p.is_zero() || is_constant(p) {
            return;
        }
        if let Some(e) = self.param_expr(p) {
            crate::solver::assume_nonzero(e);
        }
    }
}

/// Is `p` a constant (no indeterminate at a positive power)?
fn is_constant(p: &GbPoly) -> bool {
    p.terms.keys().all(|e| e.iter().all(|&x| x == 0))
}

/// Does `p` involve slot `i`?
fn mentions_slot(p: &GbPoly, i: usize) -> bool {
    p.terms.keys().any(|e| e.get(i).copied().unwrap_or(0) > 0)
}

fn mono(exp: Vec<u32>, coeff: Rational) -> GbPoly {
    GbPoly::monomial(exp, coeff)
}

/// `1 − v²` for the indeterminate in slot `i`, or `1 + v²` when `plus`.
fn one_pm_square(i: usize, n: usize, plus: bool) -> GbPoly {
    let mut e = vec![0u32; n];
    e[i] = 2;
    let sq = mono(e, Rational::from(if plus { 1 } else { -1 }));
    GbPoly::constant(Rational::from(1), n).add(&sq)
}

fn gb_pow(p: &GbPoly, k: u32) -> GbPoly {
    let mut acc = GbPoly::constant(Rational::from(1), p.n_vars);
    for _ in 0..k {
        acc = acc.mul(p);
    }
    acc
}

fn binom(n: u32, k: u32) -> Integer {
    Integer::from(Integer::binomial_u(n, k))
}

/// `p ≡ A + V·B` modulo `S² + C² = 1`, where `V` is the indeterminate in slot
/// `elim` (C = 1, S = 0) and neither `A` nor `B` involves it.
fn reduce_on_circle(p: &GbPoly, elim: usize) -> (GbPoly, GbPoly) {
    let n = p.n_vars;
    let other = 1 - elim;
    let factor = one_pm_square(other, n, false);
    let mut a = GbPoly::zero(n);
    let mut b = GbPoly::zero(n);
    for (exp, coeff) in &p.terms {
        let j = exp[elim];
        let mut base = exp.clone();
        base[elim] = 0;
        let term = mono(base, coeff.clone()).mul(&gb_pow(&factor, j / 2));
        if j % 2 == 0 {
            a = a.add(&term);
        } else {
            b = b.add(&term);
        }
    }
    (a, b)
}

/// Move slot `from` to slot 0 (slot `from` must be the only one of the two in
/// use).
fn move_to_slot0(p: &GbPoly, from: usize) -> GbPoly {
    let mut out = GbPoly::zero(p.n_vars);
    for (exp, coeff) in &p.terms {
        let mut e = exp.clone();
        e[0] = exp[from];
        if from != 0 {
            e[from] = 0;
        }
        out = out.add(&mono(e, coeff.clone()));
    }
    out
}

/// The coefficient of `v^d` (slot 0) as a polynomial in the parameters.
fn slot0_coeff(p: &GbPoly, d: u32) -> GbPoly {
    let mut out = GbPoly::zero(p.n_vars);
    for (exp, coeff) in &p.terms {
        if exp[0] == d && exp[1] == 0 {
            let mut e = exp.clone();
            e[0] = 0;
            out = out.add(&mono(e, coeff.clone()));
        }
    }
    out
}

/// Total degree in `S, C` of a term.
fn sc_degree(exp: &[u32]) -> u32 {
    exp[0] + exp[1]
}

/// Integer-scale `p` and factor it over ℤ; `None` when FLINT declines.
fn factor_over_q(p: &GbPoly, vars: &[ExprId]) -> Option<Vec<GbPoly>> {
    let mut den = Integer::from(1);
    for c in p.terms.values() {
        den.lcm_mut(c.denom());
    }
    let mut terms = crate::poly::multipoly::TermMap::new();
    for (exp, c) in &p.terms {
        let v = Rational::from(c * &den);
        let (num, d) = v.into_numer_denom();
        if d != 1 {
            return None;
        }
        let mut e = exp.clone();
        while e.last() == Some(&0) {
            e.pop();
        }
        terms.insert(e, num);
    }
    let mp = crate::poly::MultiPoly {
        vars: vars.to_vec(),
        terms,
    };
    let (_, factors) = mp.factor_irreducible()?;
    let n = p.n_vars;
    Some(
        factors
            .into_iter()
            .map(|(f, _)| {
                let mut g = GbPoly::zero(n);
                for (exp, c) in f.terms {
                    let mut e = exp;
                    e.resize(n, 0);
                    g = g.add(&mono(e, Rational::from(c)));
                }
                g
            })
            .collect(),
    )
}

/// Where a candidate root lies.
enum Place {
    /// It mentions a free parameter.
    Symbolic,
    /// Provably not real.
    NonReal,
    /// Real, enclosed in `[lo, hi]`.
    Real { lo: f64, hi: f64 },
    /// Not decided.
    Undecided,
}

fn is_closed(e: ExprId, pool: &ExprPool) -> bool {
    crate::poly::collect_free_vars(e, pool)
        .iter()
        .all(|&s| crate::eval::symbols::is_named_constant(s, pool))
}

fn real_ball(e: ExprId, pool: &ExprPool) -> Option<(f64, f64)> {
    let b = IntervalEval::new(BALL_PREC).eval(e, pool)?;
    let (lo, hi) = (b.lo().to_f64(), b.hi().to_f64());
    (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
}

fn place(r: ExprId, pool: &ExprPool) -> Place {
    if !is_closed(r, pool) {
        return Place::Symbolic;
    }
    let mut ev = CBallEval::default();
    if let Ok(b) = ev.eval(r, pool) {
        if !b.im_ball().contains(0.0) {
            return Place::NonReal;
        }
        let im = b.im_ball();
        if im.is_exact() && im.mid_f64() == 0.0 {
            let (lo, hi) = (b.re_ball().lo().to_f64(), b.re_ball().hi().to_f64());
            if lo.is_finite() && hi.is_finite() {
                return Place::Real { lo, hi };
            }
        }
    }
    match real_ball(r, pool) {
        Some((lo, hi)) => Place::Real { lo, hi },
        None => Place::Undecided,
    }
}

/// Is the closed value `e` provably equal to the rational `target`?
fn equals_rational(e: ExprId, target: i32, pool: &ExprPool) -> bool {
    if let Some(r) = as_rational(e, pool) {
        return r == target;
    }
    if !is_closed(e, pool) {
        return false;
    }
    let gap = pool.add(vec![e, pool.integer(-target)]);
    crate::matrix::zero_test::zero_status(pool, gap).is_proven_zero()
}

/// Does `r` satisfy `r² + 1 = 0` (a value `tan` never takes)? `Err` when that
/// cannot be decided.
fn is_plus_minus_i(r: ExprId, pool: &ExprPool) -> Result<bool, String> {
    if !is_closed(r, pool) {
        return Ok(false);
    }
    let e = pool.add(vec![pool.pow(r, pool.integer(2_i32)), pool.integer(1_i32)]);
    let mut ev = CBallEval::default();
    if let Ok(b) = ev.eval(e, pool) {
        if b.excludes_zero() {
            return Ok(false);
        }
    }
    if crate::matrix::zero_test::zero_status(pool, e).is_proven_zero() {
        return Ok(true);
    }
    Err(format!(
        "it cannot decide whether the root {} is ±i",
        pool.display(r)
    ))
}

/// Roots of the univariate (slot 0) polynomial `g`, thawed.
fn univariate_roots(ctx: &Ctx<'_>, g: &GbPoly) -> Result<Vec<ExprId>, String> {
    let pool = ctx.pool;
    if !mentions_slot(g, 0) {
        // A constant: no root, generically.
        if g.is_zero() {
            return Err("a factor vanishes identically".into());
        }
        ctx.assume_param_nonzero(g);
        return Ok(vec![]);
    }
    let t = pool.symbol("__trig_t__", Domain::Complex);
    let mut names = ctx.vars.clone();
    names[0] = t;
    let expr = crate::solver::gbpoly_to_expr(g, &names, pool)
        .ok_or_else(|| "a polynomial could not be rebuilt".to_string())?;
    let saved = crate::solver::take_solve_side_conditions();
    let r = crate::solver::solve_polynomial_system(vec![expr], vec![t], pool);
    let inner = crate::solver::take_solve_side_conditions();
    restore(saved);
    for c in inner {
        match c {
            SideCondition::NonZero(e) => {
                let e = simp(ctx.thaw(e), pool);
                if crate::matrix::zero_test::zero_status(pool, e).is_proven_zero() {
                    return Err(format!(
                        "a root divides by {}, which is identically zero",
                        pool.display(e)
                    ));
                }
                crate::solver::assume_nonzero(e);
            }
            other => crate::solver::record_side_condition(other),
        }
    }
    match r {
        Ok(SolutionSet::Finite(sols)) => {
            Ok(sols.iter().map(|s| simp(ctx.thaw(s[0]), pool)).collect())
        }
        Ok(SolutionSet::NoSolution) => Ok(vec![]),
        Ok(SolutionSet::Parametric(_)) => {
            Err("a polynomial in the reduced unknown has no finite root set".into())
        }
        Err(e) => Err(format!("its reduced polynomial could not be solved: {e}")),
    }
}

/// Put back hypotheses taken out of the channel.
fn restore(conds: Vec<SideCondition>) {
    for c in conds {
        match c {
            SideCondition::NonZero(e) => crate::solver::assume_nonzero(e),
            other => crate::solver::record_side_condition(other),
        }
    }
}

/// `sin θ = r` (or `cos θ = r` when `cosine`) for every root `r` of `g`.
fn sine_cosine_route(ctx: &Ctx<'_>, g: &GbPoly, cosine: bool) -> Result<Vec<Family>, String> {
    let pool = ctx.pool;
    let pi = crate::eval::symbols::pi_symbol(pool);
    let mut out = Vec::new();
    for r in univariate_roots(ctx, g)? {
        let extreme = equals_rational(r, 1, pool) || equals_rational(r, -1, pool);
        if !ctx.complex {
            match place(r, pool) {
                Place::NonReal => continue,
                Place::Real { lo, hi } => {
                    if lo > 1.0 || hi < -1.0 {
                        continue;
                    }
                    let straddles = (lo <= 1.0 && hi >= 1.0) || (lo <= -1.0 && hi >= -1.0);
                    if straddles && !extreme {
                        return Err(format!(
                            "it cannot decide whether |{}| ≤ 1",
                            pool.display(r)
                        ));
                    }
                }
                Place::Symbolic => {}
                Place::Undecided => {
                    return Err(format!(
                        "it cannot decide whether the root {} is real",
                        pool.display(r)
                    ))
                }
            }
        }
        let neg = |e: ExprId| pool.mul(vec![pool.integer(-1_i32), e]);
        let (first, second) = if cosine {
            let a = simp(pool.func("acos", vec![r]), pool);
            (a, simp(neg(a), pool))
        } else {
            let a = simp(pool.func("asin", vec![r]), pool);
            (a, simp(pool.add(vec![pi, neg(a)]), pool))
        };
        out.push(Family {
            offset: first,
            period: Period::TwoPi,
        });
        if !extreme {
            out.push(Family {
                offset: second,
                period: Period::TwoPi,
            });
        }
    }
    Ok(out)
}

/// Keep a root `r` of a polynomial in `tan θ` or `tan(θ/2)`?
fn keep_tangent_root(ctx: &Ctx<'_>, r: ExprId) -> Result<bool, String> {
    if ctx.complex {
        return Ok(!is_plus_minus_i(r, ctx.pool)?);
    }
    match place(r, ctx.pool) {
        Place::NonReal => Ok(false),
        Place::Real { .. } | Place::Symbolic => Ok(true),
        Place::Undecided => Err(format!(
            "it cannot decide whether the root {} is real",
            ctx.pool.display(r)
        )),
    }
}

/// The homogenised factor as a polynomial in `T = tan θ` (slot 0), with its
/// degree `d`, when every term's degree in `S, C` has the same parity.
fn tangent_polynomial(f: &GbPoly) -> Option<(GbPoly, u32)> {
    let mut degrees = f.terms.keys().map(|e| sc_degree(e));
    let first = degrees.next()?;
    if f.terms.keys().any(|e| sc_degree(e) % 2 != first % 2) {
        return None;
    }
    let d = f.terms.keys().map(|e| sc_degree(e)).max()?;
    let mut h = GbPoly::zero(f.n_vars);
    for (exp, coeff) in &f.terms {
        let m = (d - sc_degree(exp)) / 2;
        for l in 0..=m {
            let mut e = exp.clone();
            e[0] = exp[0] + 2 * l;
            e[1] = 0;
            let c = Rational::from(coeff * Rational::from(binom(m, l)));
            h = h.add(&mono(e, c));
        }
    }
    Some((h, d))
}

/// `tan θ = r` for every root of `h`, and `θ = π/2` when `cos θ = 0` solves
/// the factor (its `Tᵈ` coefficient vanishes).
fn tangent_route(ctx: &Ctx<'_>, h: &GbPoly, d: u32) -> Result<Vec<Family>, String> {
    let pool = ctx.pool;
    let mut out = Vec::new();
    for r in univariate_roots(ctx, h)? {
        if keep_tangent_root(ctx, r)? {
            out.push(Family {
                offset: simp(pool.func("atan", vec![r]), pool),
                period: Period::Pi,
            });
        }
    }
    let lead = slot0_coeff(h, d);
    if lead.is_zero() {
        let pi = crate::eval::symbols::pi_symbol(pool);
        out.push(Family {
            offset: pool.mul(vec![pool.rational(1, 2), pi]),
            period: Period::Pi,
        });
    } else {
        ctx.assume_param_nonzero(&lead);
    }
    Ok(out)
}

/// `t = tan(θ/2)`: `S = 2t/(1+t²)`, `C = (1−t²)/(1+t²)`.
fn half_angle_route(ctx: &Ctx<'_>, f: &GbPoly) -> Result<Vec<Family>, String> {
    let pool = ctx.pool;
    let n = f.n_vars;
    let d = f.terms.keys().map(|e| sc_degree(e)).max().unwrap_or(0);
    let mut two_t_e = vec![0u32; n];
    two_t_e[0] = 1;
    let two_t = mono(two_t_e, Rational::from(2));
    let minus = one_pm_square(0, n, false);
    let plus = one_pm_square(0, n, true);
    let mut g = GbPoly::zero(n);
    let mut at_pi = GbPoly::zero(n);
    for (exp, coeff) in &f.terms {
        let (i, j) = (exp[0], exp[1]);
        let mut rest = exp.clone();
        rest[0] = 0;
        rest[1] = 0;
        let base = mono(rest.clone(), coeff.clone());
        let term = base
            .mul(&gb_pow(&two_t, i))
            .mul(&gb_pow(&minus, j))
            .mul(&gb_pow(&plus, d - i - j));
        g = g.add(&term);
        if i == 0 {
            // S = 0, C = −1.
            let sign = if j % 2 == 0 { 1 } else { -1 };
            at_pi = at_pi.add(&mono(rest, Rational::from(coeff * sign)));
        }
    }
    let mut out = Vec::new();
    for r in univariate_roots(ctx, &g)? {
        if keep_tangent_root(ctx, r)? {
            let two_atan = pool.mul(vec![pool.integer(2_i32), pool.func("atan", vec![r])]);
            out.push(Family {
                offset: simp(two_atan, pool),
                period: Period::TwoPi,
            });
        }
    }
    if at_pi.is_zero() {
        out.push(Family {
            offset: crate::eval::symbols::pi_symbol(pool),
            period: Period::TwoPi,
        });
    } else {
        ctx.assume_param_nonzero(&at_pi);
    }
    Ok(out)
}

/// Solve one irreducible factor on the circle.
fn solve_factor(ctx: &Ctx<'_>, f: &GbPoly) -> Result<Vec<Family>, String> {
    let (a, b) = reduce_on_circle(f, 1);
    if b.is_zero() {
        if !mentions_slot(&a, 0) {
            // Constant on the circle: no root unless a parameter kills it.
            ctx.assume_param_nonzero(&a);
            return Ok(vec![]);
        }
        return sine_cosine_route(ctx, &a, false);
    }
    let (a2, b2) = reduce_on_circle(f, 0);
    if b2.is_zero() {
        return sine_cosine_route(ctx, &move_to_slot0(&a2, 1), true);
    }
    if let Some((h, d)) = tangent_polynomial(f) {
        return tangent_route(ctx, &h, d);
    }
    half_angle_route(ctx, f)
}

/// The real value of a closed offset.
fn offset_value(e: ExprId, pool: &ExprPool) -> Option<f64> {
    if !is_closed(e, pool) {
        return None;
    }
    let c = crate::eval::eval_complex_f64(e, pool, &HashMap::new()).ok()?;
    (c.im.abs() <= 1e-12 * (1.0 + c.re.abs()) && c.re.is_finite()).then_some(c.re)
}

/// `a ≡ b (mod p)` to rounding.
fn congruent(a: f64, b: f64, p: f64) -> bool {
    let q = (a - b) / p;
    (q - q.round()).abs() < 1e-9
}

/// Merge `v + 2πk` and `v + π + 2πk` into `v + πk`, and drop families covered
/// by another one.
fn dedupe(fams: Vec<Family>, pool: &ExprPool) -> Vec<Family> {
    let mut items: Vec<(Family, Option<f64>)> = fams
        .into_iter()
        .map(|f| (f, offset_value(f.offset, pool)))
        .collect();
    // Merge pairs of 2π families half a period apart.
    let mut i = 0;
    while i < items.len() {
        let mut merged = false;
        if items[i].0.period == Period::TwoPi {
            if let Some(vi) = items[i].1 {
                for j in (i + 1)..items.len() {
                    if items[j].0.period != Period::TwoPi {
                        continue;
                    }
                    let Some(vj) = items[j].1 else { continue };
                    if congruent(vi + std::f64::consts::PI, vj, 2.0 * std::f64::consts::PI) {
                        let keep = if vj.abs() < vi.abs() - 1e-12 { j } else { i };
                        let fam = Family {
                            offset: items[keep].0.offset,
                            period: Period::Pi,
                        };
                        let val = items[keep].1;
                        items.remove(j);
                        items[i] = (fam, val);
                        merged = true;
                        break;
                    }
                }
            }
        }
        if !merged {
            i += 1;
        }
    }
    // Drop covered families: π-periodic ones first so they absorb the rest.
    items.sort_by_key(|(f, _)| f.period == Period::TwoPi);
    let mut kept: Vec<(Family, Option<f64>)> = Vec::new();
    for (f, v) in items {
        let covered = kept.iter().any(|(g, w)| {
            if g.period == Period::TwoPi && f.period == Period::Pi {
                return false;
            }
            match (v, w) {
                (Some(v), Some(w)) => congruent(v, *w, g.period.value()),
                _ => g.offset == f.offset && g.period == f.period,
            }
        });
        if !covered {
            kept.push((f, v));
        }
    }
    kept.into_iter().map(|(f, _)| f).collect()
}

/// Candidates `p·π/q` for a closed offset value, nearest first.
fn rational_pi_candidates(v: f64) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for q in [1_i64, 2, 3, 4, 5, 6, 8, 10, 12, 24] {
        let p = (v * q as f64 / std::f64::consts::PI).round();
        if p.abs() > 4.0 * q as f64 {
            continue;
        }
        let p = p as i64;
        if (v - p as f64 * std::f64::consts::PI / q as f64).abs() < 1e-11 {
            out.push((p, q));
            break;
        }
    }
    out
}

/// `p·π/q` as an expression.
fn rational_pi(p: i64, q: i64, pool: &ExprPool) -> ExprId {
    let pi = crate::eval::symbols::pi_symbol(pool);
    if p == 0 {
        return pool.integer(0_i32);
    }
    let r = Rational::from((Integer::from(p), Integer::from(q)));
    let (n, d) = r.into_numer_denom();
    if d == 1 && n == 1 {
        return pi;
    }
    let c = rational_to_expr(&Rational::from((n, d)), pool);
    pool.mul(vec![c, pi])
}

/// Solve `equation = 0` for `var` when it is polynomial in trigonometric
/// functions of integer multiples of one affine angle. `None` when the shape
/// does not apply.
pub(super) fn try_trig_polynomial(
    equation: ExprId,
    var: ExprId,
    complex: bool,
    pool: &ExprPool,
) -> Option<TranscendentalOutcome> {
    let mut trigs: Vec<(TrigKind, ExprId, ExprId)> = Vec::new();
    if !collect_trig_kernels(equation, var, pool, 0, &mut trigs) {
        return None;
    }
    let mut seen = std::collections::HashSet::new();
    trigs.retain(|(_, node, _)| seen.insert(*node));
    if trigs.is_empty() {
        return None;
    }

    // θ = (a₀/D)·x + β with every argument Nᵢ·θ + γᵢ.
    let mut aff = Vec::with_capacity(trigs.len());
    for &(_, _, arg) in &trigs {
        aff.push(affine_in_var(arg, var, pool)?);
    }
    let a0 = simp(aff[0].0, pool);
    if is_zero_expr(a0, pool) || contains_var(a0, var, pool, 0) {
        return None;
    }
    let neg_one = pool.integer(-1_i32);
    let inv_a0 = pool.pow(a0, neg_one);
    let mut ratios = Vec::with_capacity(aff.len());
    for &(a, _) in &aff {
        let r = as_rational(simp(pool.mul(vec![a, inv_a0]), pool), pool)?;
        if r == 0 {
            return None;
        }
        ratios.push(r);
    }
    let mut den = Integer::from(1);
    for r in &ratios {
        den.lcm_mut(r.denom());
    }
    let mut ns = Vec::with_capacity(ratios.len());
    for r in &ratios {
        let n = Rational::from(r * &den);
        let n = n.numer().to_i64()?;
        if n.abs() > MAX_MULTIPLE {
            return None;
        }
        ns.push(n);
    }
    let zero = pool.integer(0_i32);
    let inv_n0 = rational_to_expr(&Rational::from((1, ns[0])), pool);
    let beta_c = simp(pool.mul(vec![aff[0].1, inv_n0]), pool);
    let absorbed = aff.iter().zip(&ns).all(|(&(_, b), &n)| {
        let gap = pool.add(vec![b, pool.mul(vec![pool.integer(-n), beta_c])]);
        is_zero_expr(simp(gap, pool), pool)
    });
    let (beta, gammas): (ExprId, Vec<ExprId>) = if absorbed {
        (beta_c, vec![zero; aff.len()])
    } else {
        (zero, aff.iter().map(|&(_, b)| simp(b, pool)).collect())
    };
    let inv_den = rational_to_expr(&Rational::from((Integer::from(1), den.clone())), pool);
    let scale = simp(pool.mul(vec![a0, inv_den]), pool);

    // Rewrite every kernel in S = sin θ, C = cos θ.
    let s = pool.symbol("__trig_s__", Domain::Real);
    let c = pool.symbol("__trig_c__", Domain::Real);
    let mut in_sc = equation;
    for (i, &(kind, node, _)) in trigs.iter().enumerate() {
        let rep = expand_kernel(kind, ns[i], gammas[i], s, c, pool);
        in_sc = substitute_node(in_sc, node, rep, pool, 0)?;
    }
    if contains_var(in_sc, var, pool, 0) {
        return None;
    }
    let (frozen, atoms) = crate::solver::opaque::freeze_free_atoms(&[in_sc], &[s, c], pool);
    let fz = frozen[0];
    let mut vars = vec![s, c];
    for p in crate::poly::collect_free_vars(fz, pool) {
        if p != s && p != c && !vars.contains(&p) {
            vars.push(p);
        }
    }
    let cleared = crate::solver::rational::clear_denominators(fz, &vars, pool).ok()?;
    let p = cleared.numer;
    if p.terms.keys().any(|e| sc_degree(e) > MAX_DEGREE) || !mentions_any_sc(&p) {
        return None;
    }

    // From here on the shape is recognised: a failure is a stated refusal.
    let _ = crate::solver::take_solve_side_conditions();
    let ctx = Ctx {
        pool,
        vars: vars.clone(),
        thaw: atoms.iter().map(|&(sym, orig)| (sym, orig)).collect(),
        complex,
    };
    let why = |w: String| {
        refuse(format!(
            "the equation is polynomial in sin/cos of one angle, but {w}"
        ))
    };
    let (ra, rb) = reduce_on_circle(&p, 1);
    if ra.is_zero() && rb.is_zero() {
        return why("it holds for every value of the unknown (an identity)".into());
    }
    let factors = factor_over_q(&p, &vars).unwrap_or_else(|| vec![p.clone()]);
    let mut fams = Vec::new();
    for f in &factors {
        if crate::budget::interrupt_pending() {
            return why("the active budget stopped it".into());
        }
        if !mentions_any_sc(f) {
            ctx.assume_param_nonzero(f);
            continue;
        }
        match solve_factor(&ctx, f) {
            Ok(v) => fams.extend(v),
            Err(w) => return why(w),
        }
    }
    let fams = dedupe(fams, pool);

    let k = family_index(equation, pool);
    let to_x = |offset: ExprId, period: Option<Period>| -> ExprId {
        let theta = match period {
            Some(p) => pool.add(vec![offset, pool.mul(vec![p.expr(pool), k])]),
            None => offset,
        };
        simp(solve_affine(&scale, &beta, theta, pool), pool)
    };
    let at = |x: ExprId, e: ExprId| -> ExprId {
        let map: HashMap<ExprId, ExprId> = [(var, x)].into_iter().collect();
        crate::kernel::subs::subs(e, &map, pool)
    };
    let mut out = Vec::with_capacity(fams.len());
    'family: for fam in fams {
        let mut offset = fam.offset;
        if let Some(v) = offset_value(offset, pool) {
            // A tangent of the equation undefined on this family: not a root.
            let x0 = to_x(offset, None);
            for &(kind, _, arg) in &trigs {
                if kind != TrigKind::Tan {
                    continue;
                }
                let cosv = pool.func("cos", vec![at(x0, arg)]);
                match real_ball(cosv, pool) {
                    Some((lo, hi)) if lo > 0.0 || hi < 0.0 => {}
                    _ => {
                        if crate::matrix::zero_test::zero_status(pool, cosv).is_proven_zero() {
                            continue 'family;
                        }
                        return why(format!(
                            "it cannot decide whether a tangent is defined at {}",
                            pool.display(x0)
                        ));
                    }
                }
            }
            // A closed offset that is a rational multiple of π, proved by
            // substitution, is written as one.
            for (pn, qn) in rational_pi_candidates(v) {
                let cand = rational_pi(pn, qn, pool);
                if cand == offset {
                    break;
                }
                let residual = at(to_x(cand, None), equation);
                if crate::matrix::zero_test::zero_status(pool, residual).is_proven_zero() {
                    offset = cand;
                }
            }
        }
        let x = to_x(offset, Some(fam.period));
        match crate::solver::opaque::spot_check(&[equation], &[var], &[x], &[], pool) {
            crate::solver::opaque::SpotCheck::Passed => out.push(x),
            _ => {
                return why(format!(
                    "the family {} could not be confirmed by substitution",
                    pool.display(x)
                ))
            }
        }
    }
    if !out.is_empty() {
        crate::solver::record_side_condition(SideCondition::InDomain(k, Domain::Integer));
    }
    Some(TranscendentalOutcome::Solved(out))
}

/// Does `p` involve `S` or `C`?
fn mentions_any_sc(p: &GbPoly) -> bool {
    mentions_slot(p, 0) || mentions_slot(p, 1)
}

/// `cos(nθ)` and `sin(nθ)` as polynomials in `S, C` (de Moivre).
fn multiple_angle(n: i64, s: ExprId, c: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let m = n.unsigned_abs() as u32;
    let mut cos_terms = Vec::new();
    let mut sin_terms = Vec::new();
    for j in 0..=m {
        let b = binom(m, j);
        let sign = if (j / 2) % 2 == 0 { 1 } else { -1 };
        let coeff = pool.integer(b * sign);
        let term = pool.mul(vec![
            coeff,
            pool.pow(c, pool.integer(i64::from(m - j))),
            pool.pow(s, pool.integer(i64::from(j))),
        ]);
        if j % 2 == 0 {
            cos_terms.push(term);
        } else {
            sin_terms.push(term);
        }
    }
    let cos_e = sum(cos_terms, pool);
    let mut sin_e = sum(sin_terms, pool);
    if n < 0 {
        sin_e = pool.mul(vec![pool.integer(-1_i32), sin_e]);
    }
    (cos_e, sin_e)
}

/// `f(n·θ + γ)` in `S, C`.
fn expand_kernel(
    kind: TrigKind,
    n: i64,
    gamma: ExprId,
    s: ExprId,
    c: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let (cn, sn) = multiple_angle(n, s, c, pool);
    let (cos_e, sin_e) = if is_zero_expr(gamma, pool) {
        (cn, sn)
    } else {
        let cg = simp(pool.func("cos", vec![gamma]), pool);
        let sg = simp(pool.func("sin", vec![gamma]), pool);
        let m1 = pool.integer(-1_i32);
        (
            pool.add(vec![pool.mul(vec![cn, cg]), pool.mul(vec![m1, sn, sg])]),
            pool.add(vec![pool.mul(vec![sn, cg]), pool.mul(vec![cn, sg])]),
        )
    };
    match kind {
        TrigKind::Sin => sin_e,
        TrigKind::Cos => cos_e,
        TrigKind::Tan => pool.mul(vec![sin_e, pool.pow(cos_e, pool.integer(-1_i32))]),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{solve_transcendental, TranscendentalOutcome};
    use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
    use std::collections::HashMap;

    fn solved(eq: ExprId, x: ExprId, pool: &ExprPool) -> Vec<ExprId> {
        match solve_transcendental(eq, x, pool) {
            TranscendentalOutcome::Solved(v) => v,
            TranscendentalOutcome::Unsupported => panic!(
                "refused {}: {:?}",
                pool.display(eq),
                super::super::take_transcendental_refusal()
            ),
        }
    }

    /// Every member with `k ∈ [−6, 6]`, as numbers.
    fn members(sols: &[ExprId], pool: &ExprPool) -> Vec<f64> {
        let mut out = Vec::new();
        for &s in sols {
            for kv in -6..=6 {
                let mut env = HashMap::new();
                for v in crate::poly::collect_free_vars(s, pool) {
                    if pool.with(v, |d| {
                        matches!(
                            d,
                            ExprData::Symbol {
                                domain: Domain::Integer,
                                ..
                            }
                        )
                    }) {
                        env.insert(v, kv as f64);
                    }
                }
                out.push(crate::jit::eval_interp(s, &env, pool).expect("evaluates"));
            }
        }
        out
    }

    /// Members satisfy `f`, and every sign change of `f` on [−10, 10] is a
    /// member.
    fn check(sols: &[ExprId], f: impl Fn(f64) -> f64, pool: &ExprPool) {
        let ms = members(sols, pool);
        for &m in &ms {
            assert!(f(m).abs() < 1e-9, "member {m} gives {}", f(m));
        }
        let n = 200_000;
        let mut prev = f(-10.0);
        for i in 1..=n {
            let x = -10.0 + 20.0 * i as f64 / n as f64;
            let cur = f(x);
            if prev.abs() < 1.0 && cur.abs() < 1.0 && (prev < 0.0) != (cur < 0.0) {
                assert!(
                    ms.iter().any(|m| (m - x).abs() < 1e-3),
                    "root near {x} missing"
                );
            }
            prev = cur;
        }
    }

    fn setup() -> (ExprPool, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        (pool, x)
    }

    #[test]
    fn two_sin_squared_is_one() {
        let (pool, x) = setup();
        let s = pool.func("sin", vec![x]);
        let eq = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), pool.pow(s, pool.integer(2_i32))]),
            pool.integer(-1_i32),
        ]);
        let sols = solved(eq, x, &pool);
        // π/4 + πk and −π/4 + πk: the four 2π-families merge pairwise.
        assert_eq!(sols.len(), 2);
        check(&sols, |v| 2.0 * v.sin().powi(2) - 1.0, &pool);
    }

    #[test]
    fn sin_equals_cos() {
        let (pool, x) = setup();
        let eq = pool.add(vec![
            pool.func("sin", vec![x]),
            pool.mul(vec![pool.integer(-1_i32), pool.func("cos", vec![x])]),
        ]);
        let sols = solved(eq, x, &pool);
        assert_eq!(sols.len(), 1);
        check(&sols, |v| v.sin() - v.cos(), &pool);
    }

    #[test]
    fn double_angle_against_cos() {
        let (pool, x) = setup();
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let eq = pool.add(vec![
            pool.func("sin", vec![two_x]),
            pool.mul(vec![pool.integer(-1_i32), pool.func("cos", vec![x])]),
        ]);
        let sols = solved(eq, x, &pool);
        check(&sols, |v| (2.0 * v).sin() - v.cos(), &pool);
    }

    #[test]
    fn sin_cos_product() {
        let (pool, x) = setup();
        let eq = pool.add(vec![
            pool.mul(vec![pool.func("sin", vec![x]), pool.func("cos", vec![x])]),
            pool.rational(-1, 4),
        ]);
        let sols = solved(eq, x, &pool);
        assert_eq!(sols.len(), 2);
        check(&sols, |v| v.sin() * v.cos() - 0.25, &pool);
    }

    #[test]
    fn half_angle_route_and_the_point_pi() {
        // sin x + cos x = −1: x = π + 2πk and x = −π/2 + 2πk; the first is the
        // point tan(x/2) cannot reach.
        let (pool, x) = setup();
        let eq = pool.add(vec![
            pool.func("sin", vec![x]),
            pool.func("cos", vec![x]),
            pool.integer(1_i32),
        ]);
        let sols = solved(eq, x, &pool);
        assert_eq!(sols.len(), 2);
        check(&sols, |v| v.sin() + v.cos() + 1.0, &pool);
    }

    #[test]
    fn sqrt3_sin_plus_cos() {
        let (pool, x) = setup();
        let sqrt3 = pool.pow(pool.integer(3_i32), pool.rational(1, 2));
        let eq = pool.add(vec![
            pool.mul(vec![sqrt3, pool.func("sin", vec![x])]),
            pool.func("cos", vec![x]),
            pool.integer(-1_i32),
        ]);
        let sols = solved(eq, x, &pool);
        check(&sols, |v| 3f64.sqrt() * v.sin() + v.cos() - 1.0, &pool);
    }

    #[test]
    fn tangent_squared_and_mixed_tangent() {
        let (pool, x) = setup();
        let t = pool.func("tan", vec![x]);
        let eq = pool.add(vec![pool.pow(t, pool.integer(2_i32)), pool.integer(-3_i32)]);
        let sols = solved(eq, x, &pool);
        check(&sols, |v| v.tan().powi(2) - 3.0, &pool);
        // tan x = 2 sin x: sin x = 0 or cos x = 1/2.
        let eq2 = pool.add(vec![
            t,
            pool.mul(vec![pool.integer(-2_i32), pool.func("sin", vec![x])]),
        ]);
        let sols2 = solved(eq2, x, &pool);
        check(&sols2, |v| v.tan() - 2.0 * v.sin(), &pool);
    }

    #[test]
    fn identity_is_refused() {
        let (pool, x) = setup();
        let two = pool.integer(2_i32);
        let eq = pool.add(vec![
            pool.pow(pool.func("sin", vec![x]), two),
            pool.pow(pool.func("cos", vec![x]), two),
            pool.integer(-1_i32),
        ]);
        assert!(matches!(
            solve_transcendental(eq, x, &pool),
            TranscendentalOutcome::Unsupported
        ));
    }

    #[test]
    fn no_real_solution() {
        // sin²x = 2 has no real root.
        let (pool, x) = setup();
        let eq = pool.add(vec![
            pool.pow(pool.func("sin", vec![x]), pool.integer(2_i32)),
            pool.integer(-2_i32),
        ]);
        assert!(solved(eq, x, &pool).is_empty());
    }
}
