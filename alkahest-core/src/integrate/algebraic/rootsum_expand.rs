//! Turn a Rothstein–Trager `RootSum` into an explicit **real** closed form.
//!
//! # Why this exists
//!
//! An Euler substitution reduces `∫ R(x, √(quadratic)) dx` to `∫ rational(t) dt`
//! (see [`super::parametrize`]).  The rational integrator then runs
//! Rothstein–Trager, and whenever the logarithmic part has residues outside `ℚ`
//! it reports them as a binder
//!
//! ```text
//!   RootSum(m(r), r . r·log(L(t, r)))
//! ```
//!
//! — "sum this over every root of `m`".  That is a correct and compact answer,
//! but it is *unusable* downstream:
//!
//! * `kernel::subs` does not descend into the binder, so back-substituting
//!   `t ↦ t(x)` silently leaves the internal Euler symbol inside it;
//! * `simplify` treats a `RootSum` as an opaque atom, and `gate::eval_at` has no
//!   numeric rule for one, so **no** verification tier can ever accept it (the
//!   same observation that motivates `rational_integrate::RootSumSuppressed`).
//!
//! So every Euler reduction whose `t`-integral had an algebraic residue died at
//! the gate, even though the antiderivative is elementary and real.  Measured on
//! Charlwood's Fifty, that single blockage accounted for the residuals of #30
//! (`√(1−x²)/(1+x²)`) and #47 (`x(1+2x²)/(√(1+x²)(1+x²+x⁴))`), plus a whole
//! family of `∫ dx/((quadratic)·√(quadratic))` integrands.
//!
//! # What it does
//!
//! `m` is split into linear and **monic quadratic** factors, the quadratics
//! allowed radical coefficients (see [`split_factors`]).  Each factor then
//! contributes:
//!
//! * *linear*, or *quadratic with non-negative discriminant* — substitute the
//!   explicit root(s) `(−β ± √(β²−4γ))/2` into the body and add.  Real by
//!   construction.
//! * *quadratic with negative discriminant* — the two roots are a conjugate pair
//!   `p ± iq`, so their two body terms are conjugates and their sum is
//!   `2·Re body(p+iq)`.  [`ceval`] evaluates the body in `ℝ[i]/(i²+1)` with
//!   `log(A+iB) = ½log(A²+B²) + i·atan(B/A)`, and the real part is taken.  This
//!   is what produces the `atan` half of a textbook answer.
//!
//! Everything else — a cubic with no rational root, a quartic that is neither
//! biquadratic nor has a rational root, a body this evaluator cannot read —
//! returns `None`, and the caller declines.  **No path here produces a
//! `NonElementary` verdict:** failure to expand is a failure of *this* method,
//! never a proof about the integral.
//!
//! # Soundness
//!
//! Real/complex case selection is made from an `f64` evaluation of the
//! discriminant.  That is a *heuristic choice of formula*, not a proof: a
//! misclassification yields a candidate that is wrong (or complex), and the
//! caller's `d/dx F = integrand` gate rejects it.  Nothing is emitted on the
//! strength of the numeric sign alone.
//!
//! The `atan` branch is the principal one, so the emitted antiderivative can
//! jump by a constant across a zero of the real part `A`.  That is the ordinary
//! CAS convention for a real logarithmic part (the derivative, which is what the
//! gate checks, is unaffected), and it matches what the rational engine already
//! emits for a rational-residue log part.

use super::poly_utils::is_free_of;
use crate::integrate::risch::poly_rde::{degree, expr_to_qpoly, rational_to_expr, trim};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use rug::{Integer, Rational};

type QPoly = Vec<Rational>;

/// Below this an `f64` discriminant counts as a double root rather than as a
/// sign.  A misclassification only ever costs a decline at the caller's gate.
const DISC_TOL: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Rewrite every `RootSum` in `expr` into explicit real form.
///
/// Returns `Some(expr)` unchanged when there is no `RootSum`, and `None` when
/// some `RootSum` is outside the scope above — in which case the caller must
/// decline, never weaken the verdict.
pub(super) fn expand_rootsums(expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if !contains_root_sum(expr, pool) {
        return Some(expr);
    }
    let out = rewrite(expr, pool)?;
    let out = simplify(out, pool).value;
    let out = simplify(reduce_sqrt_powers(out, pool), pool).value;
    Some(simplify(collect_like_terms(out, pool), pool).value)
}

/// Sum the numeric coefficients of structurally identical terms in every `Add`.
///
/// `simplify` does not do this when the shared part is an irrational constant:
/// `8·(√3·(−1/32) + √3·(1/32))` is returned verbatim rather than as `0`
/// (verified directly — it is not a budget effect). The complex expansion
/// produces exactly that shape, because the real part of `(p+iq)³` with
/// `p = ±√3/4` is a difference of two equal `√3` multiples, and one such
/// uncancelled zero per power is what made Charlwood #47's answer unreadable.
///
/// This is `Σ cᵢ·u` → `(Σ cᵢ)·u` and nothing else: a numeric identity on any
/// `u` whatsoever, with no side condition.
fn collect_like_terms(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Add(args) => {
            let parts: Vec<ExprId> = args.iter().map(|&a| collect_like_terms(a, pool)).collect();
            let mut order: Vec<ExprId> = Vec::new();
            let mut coeffs: std::collections::HashMap<ExprId, Rational> =
                std::collections::HashMap::new();
            for part in parts {
                let (c, rest) = split_numeric_factor(part, pool);
                if !coeffs.contains_key(&rest) {
                    order.push(rest);
                }
                *coeffs.entry(rest).or_default() += c;
            }
            let mut out = Vec::new();
            for rest in order {
                let c = &coeffs[&rest];
                if *c == 0 {
                    continue;
                }
                out.push(if *c == 1 {
                    rest
                } else {
                    pool.mul(vec![rational_to_expr(c, pool), rest])
                });
            }
            match out.len() {
                0 => pool.integer(0_i32),
                1 => out[0],
                _ => pool.add(out),
            }
        }
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&a| collect_like_terms(a, pool))
                .collect::<Vec<_>>(),
        ),
        ExprData::Pow { base, exp } => pool.pow(collect_like_terms(base, pool), exp),
        ExprData::Func { ref name, ref args } => pool.func(
            name,
            args.iter()
                .map(|&a| collect_like_terms(a, pool))
                .collect::<Vec<_>>(),
        ),
        _ => expr,
    }
}

/// Split a term into its rational coefficient and the rest of the product.
fn split_numeric_factor(expr: ExprId, pool: &ExprPool) -> (Rational, ExprId) {
    let one = pool.integer(1_i32);
    match pool.get(expr) {
        ExprData::Integer(n) => (Rational::from(n.0.clone()), one),
        ExprData::Rational(r) => (r.0.clone(), one),
        ExprData::Mul(args) => {
            let mut c = Rational::from(1);
            let mut rest: Vec<ExprId> = Vec::new();
            for a in &args {
                match pool.get(*a) {
                    ExprData::Integer(n) => c *= Rational::from(n.0.clone()),
                    ExprData::Rational(r) => c *= r.0.clone(),
                    _ => rest.push(*a),
                }
            }
            // Sort so the same multiset of factors always interns to one id.
            rest.sort_by_key(|e| e.0);
            match rest.len() {
                0 => (c, one),
                1 => (c, rest[0]),
                _ => (c, pool.mul(rest)),
            }
        }
        _ => (Rational::from(1), expr),
    }
}

/// Rewrite `√u ^ n` (`n` an integer, `|n| ≥ 2`) as `u^(n/2)` or
/// `u^((n−1)/2)·√u`.
///
/// The complex expansion below multiplies a root `p + i·q` by itself, and `p`
/// and `q` are radicals — so `√3·√3` accumulates as `√3²`, which `simplify`
/// leaves alone (it will not assume the radicand is non-negative).  Left in, one
/// such power per degree turns a two-term answer into the unreadable nest
/// Charlwood #47 produced.  Reducing them here is purely cosmetic and cannot
/// change a value: `√u ^ 2 = u` holds wherever `√u` is defined at all.
fn reduce_sqrt_powers(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let b = reduce_sqrt_powers(base, pool);
            let e = reduce_sqrt_powers(exp, pool);
            let radicand = match pool.get(b) {
                ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
                    Some(args[0])
                }
                _ => None,
            };
            match (radicand, as_i64(e, pool)) {
                (Some(u), Some(n)) if n.abs() >= 2 => {
                    let (halves, odd) = (n / 2, n % 2 != 0);
                    let p = pool.pow(u, pool.integer(halves));
                    if odd {
                        pool.mul(vec![p, b])
                    } else {
                        p
                    }
                }
                _ => pool.pow(b, e),
            }
        }
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&a| reduce_sqrt_powers(a, pool))
                .collect::<Vec<_>>(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&a| reduce_sqrt_powers(a, pool))
                .collect::<Vec<_>>(),
        ),
        ExprData::Func { ref name, ref args } => pool.func(
            name,
            args.iter()
                .map(|&a| reduce_sqrt_powers(a, pool))
                .collect::<Vec<_>>(),
        ),
        _ => expr,
    }
}

/// Does `expr` contain a `RootSum` node anywhere?
pub(super) fn contains_root_sum(expr: ExprId, pool: &ExprPool) -> bool {
    fn walk(e: ExprId, pool: &ExprPool, seen: &mut std::collections::HashSet<ExprId>) -> bool {
        if !seen.insert(e) {
            return false;
        }
        let mut hit = false;
        let kids: Vec<ExprId> = pool.with(e, |data| match data {
            ExprData::RootSum { .. } => {
                hit = true;
                vec![]
            }
            ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => args.clone(),
            ExprData::Pow { base, exp } => vec![*base, *exp],
            _ => vec![],
        });
        hit || kids.iter().any(|&c| walk(c, pool, seen))
    }
    walk(expr, pool, &mut std::collections::HashSet::new())
}

fn rewrite(expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    match pool.get(expr) {
        ExprData::RootSum { poly, var, body } => expand_one(poly, var, body, pool),
        ExprData::Add(args) => {
            let mut out = Vec::with_capacity(args.len());
            for a in &args {
                out.push(rewrite(*a, pool)?);
            }
            Some(pool.add(out))
        }
        ExprData::Mul(args) => {
            let mut out = Vec::with_capacity(args.len());
            for a in &args {
                out.push(rewrite(*a, pool)?);
            }
            Some(pool.mul(out))
        }
        ExprData::Pow { base, exp } => {
            let b = rewrite(base, pool)?;
            let e = rewrite(exp, pool)?;
            Some(pool.pow(b, e))
        }
        ExprData::Func { ref name, ref args } => {
            let mut out = Vec::with_capacity(args.len());
            for a in args {
                out.push(rewrite(*a, pool)?);
            }
            Some(pool.func(name, out))
        }
        _ => Some(expr),
    }
}

/// Could [`expand_rootsums`] rewrite `RootSum(m, r, …)` into explicit real
/// form, judged from the minimal polynomial `m` alone?
///
/// This is the **decidable half** of the scope documented at the top of this
/// module.  [`split_factors`] either finds a linear / monic-quadratic
/// factorisation of `m` or it does not, and when it does not, [`expand_one`]
/// returns `None` whatever the body looks like.  The body half is *not* decided
/// here: [`eval_factor`] can still decline a body [`ceval`] cannot read.  So the
/// predicate is **necessary but not sufficient** —
///
/// * `false` ⇒ expansion is impossible, and building the `RootSum` is waste;
/// * `true`  ⇒ expansion is not ruled out by `m`, so it is worth building.
///
/// It exists for [`crate::integrate::risch::rational_integrate`], which has `m`
/// in hand at the moment it would start the expensive Lazard–Rioboo–Trager log
/// argument and can skip that work on a `false`.  It is a *cost* decision only:
/// no verdict depends on it, and a `true` that expansion later refuses costs
/// exactly the decline it costs today.
pub(crate) fn minpoly_expandable(m: &[Rational], pool: &ExprPool) -> bool {
    split_factors(&trim(m.to_vec()), pool).is_some()
}

/// Expand a single `RootSum(m(r), r . body)`.
fn expand_one(poly: ExprId, rvar: ExprId, body: ExprId, pool: &ExprPool) -> Option<ExprId> {
    // A nested `RootSum` inside the body is out of scope.
    if contains_root_sum(body, pool) {
        return None;
    }
    let m = trim(expr_to_qpoly(poly, rvar, pool)?);
    let factors = split_factors(&m, pool)?;
    let mut terms = Vec::new();
    for f in factors {
        terms.push(eval_factor(&f, rvar, body, pool)?);
    }
    Some(pool.add(terms))
}

// ---------------------------------------------------------------------------
// Factoring `m` into linear and monic-quadratic pieces
// ---------------------------------------------------------------------------

/// One piece of `m`: a known root, or a monic `r² + β·r + γ` whose coefficients
/// may be radicals.
///
/// `disc` carries `β² − 4γ` **explicitly** rather than leaving it to be formed
/// and simplified from `β`.  On the biquadratic split `β = ±√(2√q − p)`, and
/// `simplify` does not reduce `√(3/4)²` back to `3/4`, so a derived discriminant
/// arrives as an unreduced tower of nested radicals and drags that tower through
/// every root, every `atan` argument and every `log` argument of the answer.
/// Supplied at construction the same discriminant is `−2√q − p` — one radical,
/// no nesting.
enum Factor {
    Lin(ExprId),
    Quad {
        beta: ExprId,
        gamma: ExprId,
        disc: ExprId,
    },
}

/// Split `m` into linear and monic-quadratic factors, allowing radical
/// coefficients on the quadratics.
///
/// Handles: degree ≤ 2 directly; any degree by peeling rational roots; and the
/// **biquadratic** `r⁴ + p·r² + q` left after peeling, which always splits into
/// two monic quadratics —
/// * `p² − 4q ≥ 0`: `(r² − z₊)(r² − z₋)` with `z± = (−p ± √(p²−4q))/2`;
/// * `p² − 4q < 0` (and `q > 0`): `(r² − s·r + √q)(r² + s·r + √q)` with
///   `s = √(2√q − p)`, the classical resolvent of a biquadratic with complex
///   `r²`.  This is the shape Charlwood #47 reaches (`r⁴ − r²/4 + 1/16`, whose
///   roots `±√3/4 ± i/4` give the `√3`-log plus `atan` textbook answer).
///
/// Anything else returns `None`.
fn split_factors(m: &QPoly, pool: &ExprPool) -> Option<Vec<Factor>> {
    let d = degree(m);
    if d < 1 {
        // A constant `m` has no roots; the sum is empty.
        return Some(Vec::new());
    }
    let monic = {
        let lead = m.last()?.clone();
        if lead == 0 {
            return None;
        }
        m.iter()
            .map(|c| c.clone() / lead.clone())
            .collect::<QPoly>()
    };
    match d {
        1 => Some(vec![Factor::Lin(rational_to_expr(
            &(-monic[0].clone()),
            pool,
        ))]),
        2 => {
            let d = monic[1].clone() * monic[1].clone() - Rational::from(4) * monic[0].clone();
            Some(vec![Factor::Quad {
                beta: rational_to_expr(&monic[1], pool),
                gamma: rational_to_expr(&monic[0], pool),
                disc: rational_to_expr(&d, pool),
            }])
        }
        _ => {
            // Peel one rational root and recurse; the RootSum resultants that
            // reach here are usually irreducible, so this rarely fires, but it
            // keeps a reducible `m` from being thrown away.
            if let Some(root) = rational_root(&monic) {
                let quotient = synthetic_divide(&monic, &root);
                let mut rest = split_factors(&quotient, pool)?;
                rest.push(Factor::Lin(rational_to_expr(&root, pool)));
                return Some(rest);
            }
            if d == 4 && monic[3] == 0 && monic[1] == 0 {
                return biquadratic(&monic[2], &monic[0], pool);
            }
            None
        }
    }
}

/// Split the monic biquadratic `r⁴ + p·r² + q` into two monic quadratics.
fn biquadratic(p: &Rational, q: &Rational, pool: &ExprPool) -> Option<Vec<Factor>> {
    let zero = pool.integer(0_i32);
    let dz = p.clone() * p.clone() - Rational::from(4) * q.clone();
    if dz >= 0 {
        // `r² = z±` real: two quadratics `r² − z±`, whose own discriminants are
        // `4z±` (no nesting: `β = 0`).
        let root = sqrt_expr_of(&dz, pool);
        let half = pool.rational(1_i32, 2_i32);
        let neg_p = rational_to_expr(&-p.clone(), pool);
        let neg_root = neg_expr(root, pool);
        let z_hi = simplify(pool.mul(vec![half, pool.add(vec![neg_p, root])]), pool).value;
        let z_lo = simplify(pool.mul(vec![half, pool.add(vec![neg_p, neg_root])]), pool).value;
        let neg = |e: ExprId| pool.mul(vec![pool.integer(-1_i32), e]);
        let four = |e: ExprId| pool.mul(vec![pool.integer(4_i32), e]);
        return Some(vec![
            Factor::Quad {
                beta: zero,
                gamma: neg(z_hi),
                disc: four(z_hi),
            },
            Factor::Quad {
                beta: zero,
                gamma: neg(z_lo),
                disc: four(z_lo),
            },
        ]);
    }
    // `r²` complex: `r⁴+p r²+q = (r²+√q)² − (2√q − p)r²`, which needs `q > 0`
    // (so `√q` is real) and then automatically `2√q − p > 0` because
    // `p² < 4q ⇒ |p| < 2√q`.
    if *q <= 0 {
        return None;
    }
    let m = sqrt_expr_of(q, pool); // √q
    let two_m = pool.mul(vec![pool.integer(2_i32), m]);
    let neg_p = rational_to_expr(&-p.clone(), pool);
    let s2 = simplify(pool.add(vec![two_m, neg_p]), pool).value;
    let s = sqrt_of(s2, pool);
    let neg_s = neg_expr(s, pool);
    // `β² − 4γ = (2√q − p) − 4√q = −2√q − p`, stated directly so no `√(…)²`
    // survives into the answer.
    let disc = simplify(
        pool.add(vec![pool.mul(vec![pool.integer(-2_i32), m]), neg_p]),
        pool,
    )
    .value;
    Some(vec![
        Factor::Quad {
            beta: neg_s,
            gamma: m,
            disc,
        },
        Factor::Quad {
            beta: s,
            gamma: m,
            disc,
        },
    ])
}

/// `√r` as an expression: exact when `r` is a rational square, otherwise with
/// every square factor pulled out (`√(3/4) → ½√3`, not `√(3/4)`).
fn sqrt_expr_of(r: &Rational, pool: &ExprPool) -> ExprId {
    if let Some(e) = exact_sqrt(r) {
        return rational_to_expr(&e, pool);
    }
    if *r > 0 {
        // √(n/d) = √(n·d)/d, then split `n·d = k²·f` with `f` squarefree.
        let (n, d) = (r.numer().clone(), r.denom().clone());
        let prod = n * d.clone();
        let (k, f) = extract_square(&prod);
        if k != 1 {
            let coeff = Rational::from((k, d));
            return pool.mul(vec![
                rational_to_expr(&coeff, pool),
                pool.func("sqrt", vec![rational_to_expr(&Rational::from(f), pool)]),
            ]);
        }
    }
    pool.func("sqrt", vec![rational_to_expr(r, pool)])
}

/// `√e` with a rational `e` routed through [`sqrt_expr_of`].
fn sqrt_of(e: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(e) {
        ExprData::Integer(n) => sqrt_expr_of(&Rational::from(n.0.clone()), pool),
        ExprData::Rational(r) => sqrt_expr_of(&r.0, pool),
        _ => pool.func("sqrt", vec![e]),
    }
}

/// Write `n = k²·f` with `f` squarefree, returning `(k, f)`.  Trial division is
/// capped, so a large prime factor simply stays inside the radical.
fn extract_square(n: &Integer) -> (Integer, Integer) {
    /// Trial division ceiling; beyond it the remaining cofactor stays under the
    /// radical (a cosmetic loss, never a wrong value).
    const TRIAL_MAX: u32 = 10_000;
    let mut rest = n.clone();
    let mut k = Integer::from(1);
    let mut d: u32 = 2;
    while d <= TRIAL_MAX {
        let dd = Integer::from(d) * Integer::from(d);
        if dd > rest {
            break;
        }
        while rest.clone() % dd.clone() == 0 {
            rest /= dd.clone();
            k *= Integer::from(d);
        }
        d += 1;
    }
    (k, rest)
}

fn exact_sqrt(r: &Rational) -> Option<Rational> {
    if *r < 0 {
        return None;
    }
    let (n, d) = (r.numer().clone(), r.denom().clone());
    let (ns, nr) = n.sqrt_rem(Integer::new());
    if nr != 0 {
        return None;
    }
    let (ds, dr) = d.sqrt_rem(Integer::new());
    if dr != 0 {
        return None;
    }
    Some(Rational::from((ns, ds)))
}

/// A rational root of the monic `p` by the rational-root theorem, bounded so a
/// pathological coefficient cannot make this expensive.
fn rational_root(p: &QPoly) -> Option<Rational> {
    /// Divisor enumeration stops here; beyond it we simply report no root
    /// (a decline, never a claim that none exists).
    const MAX_ABS: i64 = 4096;
    // Clear denominators to integer coefficients.
    let mut lcm = Integer::from(1);
    for c in p {
        lcm = lcm.clone().lcm(c.denom());
    }
    let ints: Vec<Integer> = p
        .iter()
        .map(|c| (c.clone() * Rational::from(lcm.clone())).numer().clone())
        .collect();
    let a0 = ints.first()?.clone();
    let an = ints.last()?.clone();
    if a0 == 0 {
        return Some(Rational::new());
    }
    let divisors = |v: &Integer| -> Vec<Integer> {
        let mut out = Vec::new();
        let Some(n) = v.clone().abs().to_i64() else {
            return out;
        };
        if n > MAX_ABS {
            return out;
        }
        for k in 1..=n {
            if n % k == 0 {
                out.push(Integer::from(k));
            }
        }
        out
    };
    for num in divisors(&a0) {
        for den in divisors(&an) {
            for sign in [1_i32, -1] {
                let cand = Rational::from((num.clone() * sign, den.clone()));
                if eval_qpoly(p, &cand) == 0 {
                    return Some(cand);
                }
            }
        }
    }
    None
}

fn eval_qpoly(p: &QPoly, x: &Rational) -> Rational {
    let mut acc = Rational::new();
    for c in p.iter().rev() {
        acc = acc * x.clone() + c.clone();
    }
    acc
}

/// Divide the monic `p` by `(r − root)` (Horner), returning the quotient.
fn synthetic_divide(p: &QPoly, root: &Rational) -> QPoly {
    let n = p.len();
    let mut q = vec![Rational::new(); n.saturating_sub(1)];
    let mut carry = Rational::new();
    for i in (1..n).rev() {
        carry = p[i].clone() + carry * root.clone();
        q[i - 1] = carry.clone();
    }
    trim(q)
}

// ---------------------------------------------------------------------------
// Evaluating the body at a factor's roots
// ---------------------------------------------------------------------------

fn eval_factor(f: &Factor, rvar: ExprId, body: ExprId, pool: &ExprPool) -> Option<ExprId> {
    match f {
        Factor::Lin(root) => Some(subst_root(body, rvar, *root, pool)),
        Factor::Quad { beta, gamma, disc } => {
            let _ = gamma;
            let disc = *disc;
            let dv = crate::integrate::gate::eval_at(disc, rvar, 0.0, pool)?;
            if !dv.is_finite() {
                return None;
            }
            let half = pool.rational(1_i32, 2_i32);
            let neg_beta = neg_expr(*beta, pool);
            if dv > DISC_TOL {
                // Two real roots: substitute each and add.
                let sd = sqrt_of(disc, pool);
                let r_hi = pool.mul(vec![half, pool.add(vec![neg_beta, sd])]);
                let neg_sd = neg_expr(sd, pool);
                let r_lo = pool.mul(vec![half, pool.add(vec![neg_beta, neg_sd])]);
                Some(pool.add(vec![
                    subst_root(body, rvar, r_hi, pool),
                    subst_root(body, rvar, r_lo, pool),
                ]))
            } else if dv < -DISC_TOL {
                // Conjugate pair `p ± iq`: the two body terms are conjugates, so
                // their sum is `2·Re body(p + iq)`.
                let p_re = simplify(pool.mul(vec![half, neg_beta]), pool).value;
                let neg_disc = simplify(neg_expr(disc, pool), pool).value;
                let q_im = simplify(pool.mul(vec![half, sqrt_of(neg_disc, pool)]), pool).value;
                let z = ceval(body, rvar, p_re, q_im, pool)?;
                Some(pool.mul(vec![pool.integer(2_i32), z.re]))
            } else {
                // Double root at `−β/2`, counted with multiplicity.
                let r0 = pool.mul(vec![half, neg_beta]);
                Some(pool.mul(vec![pool.integer(2_i32), subst_root(body, rvar, r0, pool)]))
            }
        }
    }
}

fn subst_root(body: ExprId, rvar: ExprId, root: ExprId, pool: &ExprPool) -> ExprId {
    let mut map = std::collections::HashMap::new();
    map.insert(rvar, root);
    simplify(crate::kernel::subs(body, &map, pool), pool).value
}

// ---------------------------------------------------------------------------
// Complex evaluation of the body at `r = p + i·q`
// ---------------------------------------------------------------------------

/// A value of `ℝ[i]/(i²+1)` carried as a pair of real expressions.
struct C {
    re: ExprId,
    im: ExprId,
}

fn ceval(e: ExprId, rvar: ExprId, p: ExprId, q: ExprId, pool: &ExprPool) -> Option<C> {
    if e == rvar {
        return Some(C { re: p, im: q });
    }
    if is_free_of(e, rvar, pool) {
        return Some(C {
            re: e,
            im: pool.integer(0_i32),
        });
    }
    match pool.get(e) {
        ExprData::Add(args) => {
            let mut acc = C {
                re: pool.integer(0_i32),
                im: pool.integer(0_i32),
            };
            for a in &args {
                let v = ceval(*a, rvar, p, q, pool)?;
                acc = C {
                    re: pool.add(vec![acc.re, v.re]),
                    im: pool.add(vec![acc.im, v.im]),
                };
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = C {
                re: pool.integer(1_i32),
                im: pool.integer(0_i32),
            };
            for a in &args {
                let v = ceval(*a, rvar, p, q, pool)?;
                acc = cmul(&acc, &v, pool);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            if !is_free_of(exp, rvar, pool) {
                return None;
            }
            let n = as_i64(exp, pool)?;
            let b = ceval(base, rvar, p, q, pool)?;
            cpow(&b, n, pool)
        }
        ExprData::Func { ref name, ref args } if name == "log" && args.len() == 1 => {
            let z = ceval(args[0], rvar, p, q, pool)?;
            Some(clog(&z, pool))
        }
        // Any other head applied to an `r`-dependent argument would need a
        // complex continuation we do not have; decline.
        _ => None,
    }
}

/// `−e`, pushed through a top-level sum and through a numeric factor.
///
/// `simplify` keeps `−1·(−½ + −√5/2)` as written, so a discriminant negated the
/// naive way arrives inside a `√` as `√(−1·(−½ − √5/2))` and stays that way
/// through every one of the four terms it appears in.  Negation distributing
/// over addition needs no side condition, so doing it here is free and keeps the
/// constants of a nested-radical answer legible (Charlwood #8's residual, whose
/// resultant is a biquadratic over `ℚ(√5)`, is the case that showed it).
fn neg_expr(e: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(e) {
        ExprData::Integer(n) => rational_to_expr(&-Rational::from(n.0.clone()), pool),
        ExprData::Rational(r) => rational_to_expr(&-r.0.clone(), pool),
        ExprData::Add(args) => {
            pool.add(args.iter().map(|&a| neg_expr(a, pool)).collect::<Vec<_>>())
        }
        ExprData::Mul(args) => {
            // Negate exactly one numeric factor if there is one, else prepend −1.
            let mut out: Vec<ExprId> = args.to_vec();
            if let Some(i) = out
                .iter()
                .position(|&a| matches!(pool.get(a), ExprData::Integer(_) | ExprData::Rational(_)))
            {
                out[i] = neg_expr(out[i], pool);
                pool.mul(out)
            } else {
                out.insert(0, pool.integer(-1_i32));
                pool.mul(out)
            }
        }
        _ => pool.mul(vec![pool.integer(-1_i32), e]),
    }
}

fn cmul(a: &C, b: &C, pool: &ExprPool) -> C {
    let neg1 = pool.integer(-1_i32);
    C {
        re: pool.add(vec![
            pool.mul(vec![a.re, b.re]),
            pool.mul(vec![neg1, a.im, b.im]),
        ]),
        im: pool.add(vec![pool.mul(vec![a.re, b.im]), pool.mul(vec![a.im, b.re])]),
    }
}

fn cpow(a: &C, n: i64, pool: &ExprPool) -> Option<C> {
    /// Guards against a pathological exponent turning expansion into a blow-up.
    const MAX_POW: i64 = 64;
    if n.abs() > MAX_POW {
        return None;
    }
    if n == 0 {
        return Some(C {
            re: pool.integer(1_i32),
            im: pool.integer(0_i32),
        });
    }
    if n < 0 {
        let inv = cinv(a, pool);
        return cpow(&inv, -n, pool);
    }
    let mut acc = C {
        re: pool.integer(1_i32),
        im: pool.integer(0_i32),
    };
    for _ in 0..n {
        acc = cmul(&acc, a, pool);
    }
    Some(acc)
}

/// `1/(a+bi) = (a − bi)/(a²+b²)`.
fn cinv(a: &C, pool: &ExprPool) -> C {
    let norm = pool.add(vec![
        pool.pow(a.re, pool.integer(2_i32)),
        pool.pow(a.im, pool.integer(2_i32)),
    ]);
    let inv = pool.pow(norm, pool.integer(-1_i32));
    C {
        re: pool.mul(vec![a.re, inv]),
        im: pool.mul(vec![pool.integer(-1_i32), a.im, inv]),
    }
}

/// `log(A + iB) = ½·log(A²+B²) + i·atan(B/A)` (principal branch).
fn clog(z: &C, pool: &ExprPool) -> C {
    use super::poly_utils::is_zero_expr;
    let zi = simplify(z.im, pool).value;
    if is_zero_expr(zi, pool) {
        return C {
            re: pool.func("log", vec![z.re]),
            im: pool.integer(0_i32),
        };
    }
    let norm = pool.add(vec![
        pool.pow(z.re, pool.integer(2_i32)),
        pool.pow(zi, pool.integer(2_i32)),
    ]);
    let half = pool.rational(1_i32, 2_i32);
    let ratio = pool.mul(vec![zi, pool.pow(z.re, pool.integer(-1_i32))]);
    C {
        re: pool.mul(vec![half, pool.func("log", vec![norm])]),
        im: pool.func("atan", vec![ratio]),
    }
}

fn as_i64(e: ExprId, pool: &ExprPool) -> Option<i64> {
    match pool.get(e) {
        ExprData::Integer(n) => n.0.to_i64(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::risch::poly_rde::qpoly_to_expr;
    use crate::kernel::Domain;

    /// `RootSum(r² + 1/2, r·log(t² − 4rt − 1))` is the log part Charlwood #30's
    /// Euler reduction produces.  Its roots are `±i/√2`, so the expansion must be
    /// a **real** multiple of an `atan` — with no `log` and no leftover binder.
    #[test]
    fn conjugate_pair_becomes_a_real_atan() {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let r = pool.symbol("$root$", Domain::Real);
        let m = pool.add(vec![
            pool.pow(r, pool.integer(2_i32)),
            pool.rational(1_i32, 2_i32),
        ]);
        let arg = pool.add(vec![
            pool.pow(t, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(-4_i32), r, t]),
            pool.integer(-1_i32),
        ]);
        let body = pool.mul(vec![r, pool.func("log", vec![arg])]);
        let rs = pool.root_sum(m, r, body);
        let out = expand_rootsums(rs, &pool).expect("a conjugate pair must expand");
        assert!(!contains_root_sum(out, &pool));
        let s = pool.display(out).to_string();
        assert!(s.contains("atan"), "expected an atan, got {s}");
        assert!(!s.contains("$root$"), "binder leaked into {s}");

        // `d/dt` of the expansion must equal `d/dt` of the RootSum, which `diff`
        // knows how to form even though nothing can evaluate it: check instead
        // that the expansion differentiates to the Rothstein–Trager residue sum
        // numerically, via the closed form `Σ rᵢ·(∂ₜL)/L` with `r = ±i/√2`.
        let d = simplify(crate::diff::diff(out, t, &pool).unwrap().value, &pool).value;
        for &tv in &[0.4_f64, 1.7, 3.1] {
            let got = crate::integrate::gate::eval_at(d, t, tv, &pool).unwrap();
            // Σ over r=±i/√2 of r·(2t−4r)/(t²−4rt−1), done in f64 complex by hand.
            let (a, b) = (tv * tv - 1.0, -4.0 * tv * std::f64::consts::FRAC_1_SQRT_2);
            // r₁ = i/√2, L = a + i·b ⇒ r₁/L·(2t − 4r₁) summed with conjugate
            // equals 2·Re[ i/√2 · (2t − 4i/√2)/(a+ib) ].
            let (nr, ni) = (
                2.0 * tv * 0.0 + 4.0 * std::f64::consts::FRAC_1_SQRT_2 / std::f64::consts::SQRT_2,
                2.0 * tv * std::f64::consts::FRAC_1_SQRT_2,
            );
            let den = a * a + b * b;
            let want = 2.0 * (nr * a + ni * b) / den;
            assert!(
                (got - want).abs() < 1e-9 * (1.0 + want.abs()),
                "t={tv}: expansion' = {got}, residue sum = {want}"
            );
        }
    }

    /// Real roots take the substitution path and stay `log`-shaped.
    #[test]
    fn real_roots_substitute_directly() {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let r = pool.symbol("$root$", Domain::Real);
        // r² − 1/8: roots ±1/(2√2), both real.
        let m = pool.add(vec![
            pool.pow(r, pool.integer(2_i32)),
            pool.rational(-1_i32, 8_i32),
        ]);
        let arg = pool.add(vec![
            pool.pow(t, pool.integer(2_i32)),
            pool.integer(3_i32),
            pool.mul(vec![pool.integer(-8_i32), r]),
        ]);
        let body = pool.mul(vec![r, pool.func("log", vec![arg])]);
        let rs = pool.root_sum(m, r, body);
        let out = expand_rootsums(rs, &pool).expect("real roots must expand");
        assert!(!contains_root_sum(out, &pool));
        assert!(!pool.display(out).to_string().contains("atan"));

        // d/dt must match Σ rᵢ·2t/(t²+3−8rᵢ) with r = ±1/(2√2).
        let d = simplify(crate::diff::diff(out, t, &pool).unwrap().value, &pool).value;
        for &tv in &[0.5_f64, 2.0, 4.0] {
            let got = crate::integrate::gate::eval_at(d, t, tv, &pool).unwrap();
            let r0 = 1.0 / (2.0 * std::f64::consts::SQRT_2);
            let want = r0 * 2.0 * tv / (tv * tv + 3.0 - 8.0 * r0)
                - r0 * 2.0 * tv / (tv * tv + 3.0 + 8.0 * r0);
            assert!(
                (got - want).abs() < 1e-9 * (1.0 + want.abs()),
                "t={tv}: {got} vs {want}"
            );
        }
    }

    /// A `RootSum` whose defining polynomial this module cannot split must
    /// **decline**, not return a wrong expansion.
    #[test]
    fn unsplittable_declines() {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let r = pool.symbol("$root$", Domain::Real);
        // r³ − r − 1: irreducible, no rational root, not biquadratic.
        let m = pool.add(vec![
            pool.pow(r, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-1_i32), r]),
            pool.integer(-1_i32),
        ]);
        let body = pool.mul(vec![r, pool.func("log", vec![pool.add(vec![t, r])])]);
        let rs = pool.root_sum(m, r, body);
        assert!(expand_rootsums(rs, &pool).is_none());
    }

    /// `minpoly_expandable` is what `rational_integrate` decides on, so it has
    /// to track the real capability of [`split_factors`] rather than a guess at
    /// it.  Each polynomial below is checked *both* through the predicate and
    /// through a full `expand_rootsums` of a `RootSum` built on it.
    #[test]
    fn minpoly_expandable_matches_what_expansion_can_do() {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let r = pool.symbol("$root$", Domain::Real);
        let q = |cs: &[(i32, i32)]| -> QPoly {
            cs.iter()
                .map(|&(n, d)| Rational::from((n, d)))
                .collect::<QPoly>()
        };
        // Coefficients are low-order-first.
        let cases: [(&str, QPoly, bool); 6] = [
            ("linear r − 2", q(&[(-2, 1), (1, 1)]), true),
            ("quadratic r² + 1/2", q(&[(1, 2), (0, 1), (1, 1)]), true),
            (
                "biquadratic r⁴ − r²/4 + 1/16 (Charlwood #47)",
                q(&[(1, 16), (0, 1), (-1, 4), (0, 1), (1, 1)]),
                true,
            ),
            (
                "cubic with a rational root, (r−1)(r²+1)",
                q(&[(-1, 1), (1, 1), (-1, 1), (1, 1)]),
                true,
            ),
            (
                "irreducible cubic r³ − r − 1",
                q(&[(-1, 1), (-1, 1), (0, 1), (1, 1)]),
                false,
            ),
            (
                "quartic r⁴ + r + 1: no rational root, not biquadratic",
                q(&[(1, 1), (1, 1), (0, 1), (0, 1), (1, 1)]),
                false,
            ),
        ];
        for (name, m, want) in cases {
            assert_eq!(minpoly_expandable(&m, &pool), want, "predicate on {name}");
            let m_expr = qpoly_to_expr(&m, r, &pool);
            let body = pool.mul(vec![r, pool.func("log", vec![pool.add(vec![t, r])])]);
            let rs = pool.root_sum(m_expr, r, body);
            assert_eq!(
                expand_rootsums(rs, &pool).is_some(),
                want,
                "expansion of {name} disagrees with the predicate"
            );
        }
    }

    /// An expression with no `RootSum` passes through untouched.
    #[test]
    fn passthrough_without_a_rootsum() {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let e = pool.add(vec![pool.func("log", vec![t]), pool.integer(3_i32)]);
        assert_eq!(expand_rootsums(e, &pool), Some(e));
    }

    /// The biquadratic split is exact: multiplying the two monic quadratics back
    /// together must reproduce `r⁴ − r²/4 + 1/16` (Charlwood #47's resultant).
    #[test]
    fn biquadratic_split_multiplies_back() {
        let pool = ExprPool::new();
        let p = Rational::from((-1, 4));
        let q = Rational::from((1, 16));
        let fs = biquadratic(&p, &q, &pool).expect("q>0 with complex r² must split");
        assert_eq!(fs.len(), 2);
        // (r²+β₁r+γ)(r²+β₂r+γ) = r⁴ + (β₁+β₂)r³ + (2γ+β₁β₂)r² + γ(β₁+β₂)r + γ².
        let (b1, g1) = match &fs[0] {
            Factor::Quad { beta, gamma, .. } => (*beta, *gamma),
            _ => panic!("expected a quadratic"),
        };
        let (b2, g2) = match &fs[1] {
            Factor::Quad { beta, gamma, .. } => (*beta, *gamma),
            _ => panic!("expected a quadratic"),
        };
        let ev = |e: ExprId| {
            crate::integrate::gate::eval_at(e, pool.symbol("z", Domain::Real), 0.0, &pool).unwrap()
        };
        let (b1, b2, g1, g2) = (ev(b1), ev(b2), ev(g1), ev(g2));
        assert!((b1 + b2).abs() < 1e-12, "r³ coefficient must vanish");
        assert!((g1 - g2).abs() < 1e-12);
        assert!(
            (2.0 * g1 + b1 * b2 - (-0.25)).abs() < 1e-12,
            "r² coefficient"
        );
        assert!((g1 * g2 - 0.0625).abs() < 1e-12, "constant coefficient");
    }
}
