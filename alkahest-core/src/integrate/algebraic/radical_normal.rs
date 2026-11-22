//! Normalising integrands in `ℚ(x)(√P, √Q)` to single-radical pieces.
//!
//! The standard algebraic path ([`super::integrate_via_decompose`]) needs one
//! generator `√F` with `F ∈ ℚ[x]`.  Two common shapes miss that:
//!
//! * a radical of a **rational function**, `√(N/D)` — rewritten with
//!   `w = √(N·D)` (`√(N/D) = w/D`);
//! * **two** square roots `√P`, `√Q` — `√x·(x+1)^{5/2}`, or the
//!   `x/(√(1−x²)·√(1+x²))` that integration by parts leaves behind.
//!
//! Both are handled by exact arithmetic in the biquadratic extension
//! `ℚ(x)[p, q]/(p² − P, q² − Q)` (basis `1, p, q, pq`; inverses through the
//! two conjugations).  The integrand normalises to `c₀ + c₁p + c₂q + c₃pq`, and
//! each piece is integrated on its own: the decomposition is into eigenspaces
//! of the Galois group `{p ↦ ±p, q ↦ ±q}`, so if `∫f` is elementary every
//! piece is (each is an average of conjugates of `f`), and nothing is lost by
//! splitting.  Each radical piece `c·√R` becomes `(c/D)·√F` with `F = N·D`
//! (square factors pulled out), a single polynomial radicand.
//!
//! **Branches.**  A piece's antiderivative is computed for the principal
//! `√F`; it is then rewritten with `√F := D·√P` (or `D·√P·√Q`, divided by the
//! extracted square factor), an expression that squares to `F` *identically*,
//! so the original radicals reappear exactly as the integrand wrote them.  The
//! result still goes through the numeric `d/dx F = f` gate against the
//! original integrand; a piece whose closed form silently assumed the
//! principal branch (an `asin` with no `√F` left in it, say) fails there and
//! the route declines.

use rug::Rational;

use super::super::risch::alg_field::{RatFn, RationalFunctionField};
use super::super::risch::number_field::CoeffField;
use super::super::risch::poly_rde::{degree, is_free_of_var, poly_mul, qpoly_to_expr, trim, QPoly};
use super::super::risch::rational_rde::expr_to_qrational;
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

const F: RationalFunctionField = RationalFunctionField;

/// An element `c₀ + c₁p + c₂q + c₃pq` of `ℚ(x)[p,q]/(p²−P, q²−Q)`.
#[derive(Clone, Debug)]
struct Biq([RatFn; 4]);

struct Ring {
    p: RatFn,
    q: RatFn,
}

impl Ring {
    fn constant(&self, c: RatFn) -> Biq {
        Biq([c, F.zero(), F.zero(), F.zero()])
    }
    fn gen(&self, k: usize) -> Biq {
        let mut c = [F.zero(), F.zero(), F.zero(), F.zero()];
        c[k] = F.one();
        Biq(c)
    }
    fn add(&self, a: &Biq, b: &Biq) -> Biq {
        Biq(std::array::from_fn(|i| F.add(&a.0[i], &b.0[i])))
    }
    fn mul(&self, a: &Biq, b: &Biq) -> Biq {
        let (a, b) = (&a.0, &b.0);
        let m = |x: &RatFn, y: &RatFn| F.mul(x, y);
        let (p, q) = (&self.p, &self.q);
        let pq = F.mul(p, q);
        let c0 = F.add(
            &F.add(&m(&a[0], &b[0]), &m(&m(&a[1], &b[1]), p)),
            &F.add(&m(&m(&a[2], &b[2]), q), &m(&m(&a[3], &b[3]), &pq)),
        );
        let c1 = F.add(
            &F.add(&m(&a[0], &b[1]), &m(&a[1], &b[0])),
            &m(&F.add(&m(&a[2], &b[3]), &m(&a[3], &b[2])), q),
        );
        let c2 = F.add(
            &F.add(&m(&a[0], &b[2]), &m(&a[2], &b[0])),
            &m(&F.add(&m(&a[1], &b[3]), &m(&a[3], &b[1])), p),
        );
        let c3 = F.add(
            &F.add(&m(&a[0], &b[3]), &m(&a[3], &b[0])),
            &F.add(&m(&a[1], &b[2]), &m(&a[2], &b[1])),
        );
        Biq([c0, c1, c2, c3])
    }
    fn inv(&self, a: &Biq) -> Option<Biq> {
        let conj_q = Biq([
            a.0[0].clone(),
            a.0[1].clone(),
            F.neg(&a.0[2]),
            F.neg(&a.0[3]),
        ]);
        let g = self.mul(a, &conj_q); // in ℚ(x)(p)
        let conj_p = Biq([
            g.0[0].clone(),
            F.neg(&g.0[1]),
            g.0[2].clone(),
            F.neg(&g.0[3]),
        ]);
        let norm = self.mul(&g, &conj_p); // in ℚ(x)
        if !(F.is_zero(&norm.0[1]) && F.is_zero(&norm.0[2]) && F.is_zero(&norm.0[3])) {
            return None;
        }
        let ninv = F.inv(&norm.0[0])?;
        let num = self.mul(&conj_q, &conj_p);
        Some(self.mul(&num, &self.constant(ninv)))
    }
    fn pow(&self, a: &Biq, n: i64) -> Option<Biq> {
        let base = if n < 0 { self.inv(a)? } else { a.clone() };
        let mut acc = self.constant(F.one());
        for _ in 0..n.unsigned_abs() {
            acc = self.mul(&acc, &base);
        }
        Some(acc)
    }
}

/// A square-root radicand group: the rational function `R = num/den` and one
/// original radical expression for it (`√R` as the integrand wrote it).
#[derive(Clone)]
struct Group {
    num: QPoly,
    den: QPoly,
    radical: ExprId,
}

fn same_ratfn(e: ExprId, num: &QPoly, den: &QPoly, var: ExprId, pool: &ExprPool) -> bool {
    match expr_to_qrational(e, var, pool) {
        Some((n, d)) => trim(poly_mul(&n, den)) == trim(poly_mul(num, &d)),
        None => false,
    }
}

/// Collect the `x`-dependent radicals.  `None` if any radical is not a square
/// root of a rational function, or a function other than `sqrt` wraps `x`.
fn collect(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<Group>) -> Option<()> {
    if is_free_of_var(expr, var, pool) {
        return Some(());
    }
    let push = |base: ExprId, radical: ExprId, out: &mut Vec<Group>| -> Option<()> {
        let (n, d) = expr_to_qrational(base, var, pool)?;
        let (n, d) = (trim(n), trim(d));
        if !out
            .iter()
            .any(|g| trim(poly_mul(&n, &g.den)) == trim(poly_mul(&g.num, &d)))
        {
            out.push(Group {
                num: n,
                den: d,
                radical,
            });
        }
        Some(())
    };
    match pool.get(expr) {
        ExprData::Symbol { .. } => Some(()),
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            push(args[0], expr, out)
        }
        ExprData::Func { .. } => None,
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(_) => collect(base, var, pool, out),
            ExprData::Rational(r) if *r.0.denom() == 2 => {
                let half = pool.pow(base, pool.rational(1, 2));
                push(base, half, out)
            }
            _ => None,
        },
        ExprData::Add(args) | ExprData::Mul(args) => {
            for a in args.iter() {
                collect(*a, var, pool, out)?;
            }
            Some(())
        }
        _ => None,
    }
}

fn to_biq(
    expr: ExprId,
    var: ExprId,
    groups: &[Group],
    ring: &Ring,
    pool: &ExprPool,
) -> Option<Biq> {
    if expr == var {
        return Some(ring.constant(RatFn::from_poly(&vec![
            Rational::from(0),
            Rational::from(1),
        ])));
    }
    let radical_power = |base: ExprId, k: i64| -> Option<Biq> {
        let idx = groups
            .iter()
            .position(|g| same_ratfn(base, &g.num, &g.den, var, pool))?;
        ring.pow(&ring.gen(idx + 1), k)
    };
    match pool.get(expr) {
        ExprData::Integer(n) => {
            Some(ring.constant(RatFn::from_poly(&vec![Rational::from(n.0.clone())])))
        }
        ExprData::Rational(r) => Some(ring.constant(RatFn::from_poly(&vec![r.0.clone()]))),
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            radical_power(args[0], 1)
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(m) => {
                let m = m.0.to_i64()?;
                if m.abs() > 64 {
                    return None;
                }
                let b = to_biq(base, var, groups, ring, pool)?;
                ring.pow(&b, m)
            }
            ExprData::Rational(r) if *r.0.denom() == 2 => {
                radical_power(base, r.0.numer().to_i64()?)
            }
            _ => None,
        },
        ExprData::Add(args) => {
            let mut acc = ring.constant(F.zero());
            for a in args.iter() {
                acc = ring.add(&acc, &to_biq(*a, var, groups, ring, pool)?);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = ring.constant(F.one());
            for a in args.iter() {
                acc = ring.mul(&acc, &to_biq(*a, var, groups, ring, pool)?);
            }
            Some(acc)
        }
        _ => None,
    }
}

fn ratfn_expr(r: &RatFn, var: ExprId, pool: &ExprPool) -> ExprId {
    let n = qpoly_to_expr(r.numer(), var, pool);
    if degree(r.denom()) == 0 && r.denom()[0] == 1 {
        n
    } else {
        pool.mul(vec![
            n,
            pool.pow(qpoly_to_expr(r.denom(), var, pool), pool.integer(-1_i32)),
        ])
    }
}

/// `√(num/den)` as `(s/den)·√f` with `f` squarefree: returns `(s, f)` where
/// `num·den = s²·f` (`s, f ∈ ℚ[x]`, `f` up to a rational constant factor).
fn split_square(num: &QPoly, den: &QPoly) -> Option<(QPoly, QPoly)> {
    let prod = trim(poly_mul(num, den));
    if prod.is_empty() {
        return None;
    }
    let lc = prod[prod.len() - 1].clone();
    let parts = super::super::risch::rational_integrate::yun(&prod)?;
    let mut s: QPoly = vec![Rational::from(1)];
    let mut f: QPoly = vec![lc];
    for (v, i) in parts {
        for _ in 0..i / 2 {
            s = poly_mul(&s, &v);
        }
        if i % 2 == 1 {
            f = poly_mul(&f, &v);
        }
    }
    Some((trim(s), trim(f)))
}

/// Replace every `√g`/`g^{k/2}` with `g ≡ f` (as polynomials) by `back^k`.
fn back_substitute(e: ExprId, f: &QPoly, back: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    let one: QPoly = vec![Rational::from(1)];
    match pool.get(e) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            if same_ratfn(args[0], f, &one, var, pool) {
                back
            } else {
                let a = back_substitute(args[0], f, back, var, pool);
                pool.func(name, vec![a])
            }
        }
        ExprData::Func { ref name, ref args } => {
            let a = args
                .iter()
                .map(|&a| back_substitute(a, f, back, var, pool))
                .collect();
            pool.func(name, a)
        }
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if *r.0.denom() == 2 && same_ratfn(base, f, &one, var, pool) {
                    if let Some(k) = r.0.numer().to_i32() {
                        return pool.pow(back, pool.integer(k));
                    }
                }
            }
            pool.pow(back_substitute(base, f, back, var, pool), exp)
        }
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&a| back_substitute(a, f, back, var, pool))
                .collect(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&a| back_substitute(a, f, back, var, pool))
                .collect(),
        ),
        _ => e,
    }
}

/// `log(u)` → `½·log(u²)` throughout: the same derivative, and real wherever
/// `u ≠ 0` is real (a rational antiderivative's `log` of a negative quantity
/// is otherwise complex on that interval).
fn real_logs(e: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(e) {
        ExprData::Func { ref name, ref args } if name == "log" && args.len() == 1 => {
            let inner = real_logs(args[0], pool);
            pool.mul(vec![
                pool.rational(1, 2),
                pool.func("log", vec![pool.pow(inner, pool.integer(2_i32))]),
            ])
        }
        ExprData::Func { ref name, ref args } => {
            pool.func(name, args.iter().map(|&a| real_logs(a, pool)).collect())
        }
        ExprData::Add(a) => pool.add(a.iter().map(|&c| real_logs(c, pool)).collect()),
        ExprData::Mul(a) => pool.mul(a.iter().map(|&c| real_logs(c, pool)).collect()),
        ExprData::Pow { base, exp } => pool.pow(real_logs(base, pool), exp),
        _ => e,
    }
}

/// Normalise and integrate.  `None` when the shape is out of scope (a single
/// square root of a polynomial — the standard path's job — or anything that
/// is not rational in `x` and at most two square roots of rational functions).
pub(super) fn try_radical_normalize(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let mut groups = Vec::new();
    collect(expr, var, pool, &mut groups)?;
    match groups.len() {
        1 if degree(&groups[0].den) == 0 => return None, // already standard
        1 | 2 => {}
        _ => return None,
    }
    let ratfn = |g: &Group| RatFn::new(g.num.clone(), g.den.clone());
    let ring = Ring {
        p: ratfn(&groups[0]),
        q: groups.get(1).map(ratfn).unwrap_or_else(|| F.one()),
    };
    let elem = to_biq(expr, var, &groups, &ring, pool)?;

    // Radical of each basis element: p ↦ R₀, q ↦ R₁, pq ↦ R₀R₁, with the
    // original radical expression(s) whose product squares to it.
    let mut pieces: Vec<ExprId> = Vec::new();
    let mut sign_pieces: Vec<(ExprId, ExprId)> = Vec::new();
    let mut backs: Vec<(QPoly, ExprId)> = Vec::new();
    for k in 0..4 {
        let c = &elem.0[k];
        if F.is_zero(c) {
            continue;
        }
        if k == 0 {
            pieces.push(ratfn_expr(c, var, pool));
            continue;
        }
        let (r, orig): (RatFn, ExprId) = match k {
            1 => (ring.p.clone(), groups[0].radical),
            2 => (ring.q.clone(), groups.get(1)?.radical),
            _ => (
                F.mul(&ring.p, &ring.q),
                pool.mul(vec![groups[0].radical, groups.get(1)?.radical]),
            ),
        };
        let (s, f) = split_square(r.numer(), r.denom())?;
        if degree(&f) < 1 {
            // A perfect square up to the constant κ = f: √R = σ·s/den with
            // σ := orig·den/s, σ² = κ, locally constant (it is the sign the
            // integrand's own radicals pick, e.g. `x/|x|`).  So the piece
            // integrates to σ·∫ c·s/den — written with `orig` itself, so the
            // sign is the integrand's on every interval.
            let den_poly = r.denom().clone();
            let rat = F.mul(c, &RatFn::new(s.clone(), den_poly.clone()));
            let sigma = pool.mul(vec![orig, ratfn_expr(&RatFn::new(den_poly, s), var, pool)]);
            sign_pieces.push((ratfn_expr(&rat, var, pool), sigma));
            continue;
        }
        // √R = (s/den)·√f  ⇒  piece = c·(s/den)·√f, and √f := orig·den/s.
        let den_poly = r.denom().clone();
        let coeff = F.mul(c, &RatFn::new(s.clone(), den_poly.clone()));
        let f_expr = qpoly_to_expr(&f, var, pool);
        pieces.push(pool.mul(vec![
            ratfn_expr(&coeff, var, pool),
            pool.func("sqrt", vec![f_expr]),
        ]));
        let back = simplify(
            pool.mul(vec![orig, ratfn_expr(&RatFn::new(den_poly, s), var, pool)]),
            pool,
        )
        .value;
        backs.push((f, back));
    }
    if pieces.is_empty() && sign_pieces.is_empty() {
        return None;
    }

    let mut total = Vec::new();
    for piece in pieces {
        let piece = simplify(piece, pool).value;
        let r = crate::integrate::engine::integrate(piece, var, pool).ok()?;
        total.push(r.value);
    }
    let mut f = match total.len() {
        0 => pool.integer(0_i32),
        1 => total[0],
        _ => pool.add(total),
    };
    for (fpoly, back) in &backs {
        f = back_substitute(f, fpoly, *back, var, pool);
    }
    for (rat, sigma) in sign_pieces {
        let r = crate::integrate::engine::integrate(simplify(rat, pool).value, var, pool).ok()?;
        // σ flips sign between components, so the rational antiderivative
        // must be real on all of them: `log u` → `½·log(u²)`.
        let g = real_logs(r.value, pool);
        f = pool.add(vec![f, pool.mul(vec![sigma, g])]);
    }
    let f = simplify(f, pool).value;
    let one = pool.integer(1_i32);
    if !super::parametrize::verify_derivative_pub(f, expr, one, var, pool) {
        return None;
    }
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("algebraic_radical_normalize", expr, f));
    Some(Ok(DerivedExpr::with_log(f, log)))
}

// ---------------------------------------------------------------------------
// Rationalising the algebraic cofactor of a transcendental integrand
// ---------------------------------------------------------------------------

/// Does `e` contain a square root (or half-integer power) of something that
/// depends on `var` inside a negative power — a radical in a denominator?
fn radical_in_denominator(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    fn has_radical(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
        match pool.get(e) {
            ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
                !is_free_of_var(args[0], var, pool)
            }
            ExprData::Pow { base, exp } => {
                matches!(pool.get(exp), ExprData::Rational(_)) && !is_free_of_var(base, var, pool)
                    || has_radical(base, var, pool)
            }
            ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
                a.iter().any(|&c| has_radical(c, var, pool))
            }
            _ => false,
        }
    }
    match pool.get(e) {
        ExprData::Pow { base, exp } => {
            let neg = match pool.get(exp) {
                ExprData::Integer(n) => n.0 < 0,
                ExprData::Rational(r) => r.0 < 0,
                _ => false,
            };
            (neg && has_radical(base, var, pool)) || radical_in_denominator(base, var, pool)
        }
        ExprData::Add(a) | ExprData::Mul(a) => {
            a.iter().any(|&c| radical_in_denominator(c, var, pool))
        }
        _ => false,
    }
}

/// Is `e` built from `var`, rational constants, `+ · ^integer` and square
/// roots only (no other function of `var`)?
fn is_algebraic_factor(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let mut groups = Vec::new();
    collect(e, var, pool, &mut groups).is_some()
}

/// Rewrite `T(x)·A(x)` — `T` transcendental, `A` algebraic with a radical in a
/// denominator — as `Σ T·cₖ·radₖ` with `A = c₀ + c₁√P + c₂√Q + c₃√P√Q`
/// normalised exactly (no radical left in any denominator).  `asin(x)/(1 +
/// √(1−x²))` becomes `asin(x)/x² − asin(x)·√(1−x²)/x²`, both by-parts-able.
/// The original radical expressions are reused, so the rewrite is an identity
/// on every branch.  `None` when there is nothing to rationalise.
pub(crate) fn rationalize_algebraic_cofactor(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let args = match pool.get(expr) {
        ExprData::Mul(a) => a.to_vec(),
        _ => return None,
    };
    let (alg, other): (Vec<ExprId>, Vec<ExprId>) = args
        .into_iter()
        .partition(|&a| is_algebraic_factor(a, var, pool));
    if other.iter().all(|&o| is_free_of_var(o, var, pool)) || alg.is_empty() {
        return None; // purely algebraic: the algebraic engine's job
    }
    let a = pool.mul(alg);
    if !radical_in_denominator(a, var, pool) {
        return None;
    }
    let mut groups = Vec::new();
    collect(a, var, pool, &mut groups)?;
    if groups.is_empty() || groups.len() > 2 {
        return None;
    }
    let ratfn = |g: &Group| RatFn::new(g.num.clone(), g.den.clone());
    let ring = Ring {
        p: ratfn(&groups[0]),
        q: groups.get(1).map(ratfn).unwrap_or_else(|| F.one()),
    };
    let elem = to_biq(a, var, &groups, &ring, pool)?;
    let t = pool.mul(other);
    let mut terms = Vec::new();
    for k in 0..4 {
        let c = &elem.0[k];
        if F.is_zero(c) {
            continue;
        }
        let rad = match k {
            0 => None,
            1 => Some(groups[0].radical),
            2 => Some(groups.get(1)?.radical),
            _ => Some(pool.mul(vec![groups[0].radical, groups.get(1)?.radical])),
        };
        let mut fs = vec![t, ratfn_expr(c, var, pool)];
        fs.extend(rad);
        terms.push(pool.mul(fs));
    }
    match terms.len() {
        0 => None,
        1 => Some(terms[0]),
        _ => Some(pool.add(terms)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn integrate_ok(src: &str) -> Option<ExprId> {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = std::collections::HashMap::from([("x".to_owned(), x)]);
        let e = crate::parse::parse(src, &pool, &mut syms).ok()?;
        try_radical_normalize(e, x, &pool)?.ok().map(|d| d.value)
    }

    #[test]
    fn two_linear_radicals_merge() {
        assert!(integrate_ok("sqrt(x)*(x+1)^(5/2)").is_some());
    }

    #[test]
    fn pullback_after_merge() {
        // x/(√(1−x²)·√(1+x²)) = x/√(1−x⁴) → ½·asin(x²).
        assert!(integrate_ok("x/(sqrt(1-x^2)*sqrt(1+x^2))").is_some());
    }

    #[test]
    fn rational_radicand() {
        assert!(
            integrate_ok("sqrt((x+1)/(x^2+3))*(x^2+3)").is_some()
                || integrate_ok("sqrt(x/(x+1))").is_some()
        );
    }

    #[test]
    fn single_polynomial_radical_is_not_ours() {
        assert!(integrate_ok("sqrt(x^2+1)").is_none());
    }
}
