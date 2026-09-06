//! Partial-fraction decomposition over **ℚ(params)** — the coefficient field of
//! rational functions in every symbol *other* than the decomposition variable.
//!
//! [`super::partial_fractions::apart`] decomposes over ℚ, so `1/((s+a)(s+b))`
//! is not a rational function it can see at all: `a` and `b` are not rational
//! numbers.  This module treats every non-`var` generator as a transcendental
//! constant, which turns `K = ℚ(params)` into an ordinary field and
//! `K[var]` into an ordinary Euclidean domain.  The algorithm is then the same
//! one `partial_fractions` runs over ℚ (Bronstein 2005, §1.5), with
//! [`RationalFunction`] standing in for `rug::Rational`:
//!
//! 1. Write `expr = num/den` over the generator set `{var} ∪ params`
//!    (`cancel::rational_over_generators`), already reduced to lowest terms.
//! 2. Factor `den` as a **multivariate** ℤ-polynomial.  By Gauss's lemma the
//!    irreducible factors that involve `var` are exactly the irreducible
//!    factors of `den` in `K[var]`, and the ones free of `var` are its content
//!    in `ℚ[params]`.
//! 3. Long-divide to peel the polynomial part, CRT-split over the pairwise
//!    coprime moduli `f_i^{e_i}`, then `f_i`-adically expand each part.
//!
//! # Genericity is reported, never assumed
//!
//! Step 3 divides by elements of `K`.  Every such element is a ratio of
//! parameter polynomials, and a division that is legal in the *field* `ℚ(params)`
//! can still be a division by zero at a particular parameter *value*:
//!
//! ```text
//!     1/((s+ka)(s+ke))  =  1/((ke−ka)(s+ka))  −  1/((ke−ka)(s+ke))
//! ```
//!
//! is an identity of rational functions and is **wrong** at `ka = ke`, where
//! the left side is the perfectly ordinary `1/(s+ka)²` and the right side is
//! `0/0`.  So the decomposition is returned together with the hypotheses it
//! rests on, as [`SideCondition::NonZero`] facts: the denominators actually
//! introduced, factored into irreducibles, minus the ones that already divide
//! the *input's* denominator (where the input has a pole anyway, so no new
//! hypothesis is created).
//!
//! An empty condition list means every division performed was by a non-zero
//! rational constant — not that nothing was checked.

use std::collections::BTreeMap;

use crate::deriv::SideCondition;
use crate::kernel::{ExprId, ExprPool};

use super::cancel::rational_over_generators;
use super::multipoly::MultiPoly;
use super::partial_fractions::ApartError;
use super::rational::{mpoly_exact_div, RationalFunction};

/// A coefficient of the main variable: an element of `K = ℚ(params)`.
type K = RationalFunction;

/// A polynomial in the main variable over `K`, dense, index = degree.
type KPoly = Vec<K>;

/// Same shape as `multipoly`'s private `TermMap`: exponent key → ℤ coefficient.
type TermMap = BTreeMap<Vec<u32>, rug::Integer>;

/// Guard against a parametric decomposition whose cost is not worth paying.
///
/// Every `K` operation runs a multivariate GCD through FLINT, so the Euclidean
/// algorithm over `K[var]` is far more expensive per step than the ℚ one.  The
/// inverse-transform tables top out at degree 4 (a repeated irreducible
/// quadratic); this ceiling is well clear of that and keeps a pathological
/// input from turning into an unbounded amount of work with no budget check.
///
/// Exceeding it reports [`ApartError::NotRational`], which is not a fudge:
/// every input that reaches this module has *already* been refused by the ℚ
/// decomposition for exactly that reason, and the ℚ(params) path is an
/// additional attempt on top of that refusal.  Declining to make the attempt
/// leaves the original, accurate verdict standing.
const MAX_PARAM_DEGREE: u32 = 16;

/// The decomposition plus the hypotheses it rests on.
#[derive(Debug, Clone)]
pub(crate) struct ParamApart {
    pub expr: ExprId,
    pub conditions: Vec<SideCondition>,
}

// ===========================================================================
// K = ℚ(params)
// ===========================================================================

fn k_zero(pv: &[ExprId]) -> K {
    RationalFunction {
        numer: MultiPoly::zero(pv.to_vec()),
        denom: MultiPoly::constant(pv.to_vec(), 1),
    }
}

fn k_one(pv: &[ExprId]) -> K {
    RationalFunction {
        numer: MultiPoly::constant(pv.to_vec(), 1),
        denom: MultiPoly::constant(pv.to_vec(), 1),
    }
}

fn k_from_mpoly(p: MultiPoly) -> Result<K, ApartError> {
    let one = MultiPoly::constant(p.vars.clone(), 1);
    RationalFunction::new(p, one).map_err(|_| ApartError::NotRational)
}

fn k_add(a: &K, b: &K) -> Result<K, ApartError> {
    (a.clone() + b.clone()).map_err(|_| ApartError::NotRational)
}

fn k_sub(a: &K, b: &K) -> Result<K, ApartError> {
    (a.clone() - b.clone()).map_err(|_| ApartError::NotRational)
}

fn k_mul(a: &K, b: &K) -> Result<K, ApartError> {
    (a.clone() * b.clone()).map_err(|_| ApartError::NotRational)
}

fn k_div(a: &K, b: &K) -> Result<K, ApartError> {
    if b.is_zero() {
        return Err(ApartError::ZeroDenominator);
    }
    (a.clone() / b.clone()).map_err(|_| ApartError::NotRational)
}

// ===========================================================================
// K[var]
// ===========================================================================

fn k_trim(mut p: KPoly) -> KPoly {
    while p.last().is_some_and(|c| c.is_zero()) {
        p.pop();
    }
    p
}

/// Degree, or `-1` for the zero polynomial.
fn k_degree(p: &KPoly) -> isize {
    let t = k_trim(p.clone());
    t.len() as isize - 1
}

fn kp_zero() -> KPoly {
    Vec::new()
}

fn kp_one(pv: &[ExprId]) -> KPoly {
    vec![k_one(pv)]
}

fn kp_add(a: &KPoly, b: &KPoly, pv: &[ExprId]) -> Result<KPoly, ApartError> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = a.get(i).cloned().unwrap_or_else(|| k_zero(pv));
        let y = b.get(i).cloned().unwrap_or_else(|| k_zero(pv));
        out.push(k_add(&x, &y)?);
    }
    Ok(k_trim(out))
}

fn kp_sub(a: &KPoly, b: &KPoly, pv: &[ExprId]) -> Result<KPoly, ApartError> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = a.get(i).cloned().unwrap_or_else(|| k_zero(pv));
        let y = b.get(i).cloned().unwrap_or_else(|| k_zero(pv));
        out.push(k_sub(&x, &y)?);
    }
    Ok(k_trim(out))
}

/// Equality **by value**, not by representation.
///
/// [`RationalFunction`]'s derived `PartialEq` compares numerator and
/// denominator, and a zero numerator keeps whatever denominator it was built
/// with: `0/(a−b)²` and `0/1` are the same element of `K` and compare unequal.
/// Subtracting and asking `is_zero` is representation-independent and still
/// exact.
fn kp_eq(a: &KPoly, b: &KPoly, pv: &[ExprId]) -> Result<bool, ApartError> {
    Ok(k_trim(kp_sub(a, b, pv)?).is_empty())
}

fn kp_mul(a: &KPoly, b: &KPoly, pv: &[ExprId]) -> Result<KPoly, ApartError> {
    let a = k_trim(a.clone());
    let b = k_trim(b.clone());
    if a.is_empty() || b.is_empty() {
        return Ok(kp_zero());
    }
    let mut out = vec![k_zero(pv); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        for (j, y) in b.iter().enumerate() {
            if y.is_zero() {
                continue;
            }
            let prod = k_mul(x, y)?;
            out[i + j] = k_add(&out[i + j], &prod)?;
        }
    }
    Ok(k_trim(out))
}

fn kp_scale(p: &KPoly, c: &K) -> Result<KPoly, ApartError> {
    let mut out = Vec::with_capacity(p.len());
    for x in p {
        out.push(k_mul(x, c)?);
    }
    Ok(k_trim(out))
}

fn kp_pow(p: &KPoly, n: u32, pv: &[ExprId]) -> Result<KPoly, ApartError> {
    let mut acc = kp_one(pv);
    for _ in 0..n {
        acc = kp_mul(&acc, p, pv)?;
    }
    Ok(acc)
}

/// Euclidean division in `K[var]`: `a = q·b + r` with `deg r < deg b`.
fn kp_divrem(a: &KPoly, b: &KPoly, pv: &[ExprId]) -> Result<(KPoly, KPoly), ApartError> {
    let b = k_trim(b.clone());
    let db = k_degree(&b);
    if db < 0 {
        return Err(ApartError::ZeroDenominator);
    }
    let mut r = k_trim(a.clone());
    let dr0 = k_degree(&r);
    if dr0 < db {
        return Ok((kp_zero(), r));
    }
    let lc_inv = k_div(&k_one(pv), &b[db as usize])?;
    let mut q = vec![k_zero(pv); (dr0 - db + 1) as usize];
    loop {
        let dr = k_degree(&r);
        if dr < db {
            break;
        }
        let shift = (dr - db) as usize;
        let c = k_mul(&r[dr as usize], &lc_inv)?;
        q[shift] = k_add(&q[shift], &c)?;
        // r ← r − c·var^shift·b.  The leading term cancels exactly (field
        // arithmetic is exact), so clear it outright rather than relying on
        // a subtraction to land on the canonical zero.
        for (i, bi) in b.iter().enumerate() {
            if i as isize == db {
                continue;
            }
            let d = k_mul(&c, bi)?;
            r[i + shift] = k_sub(&r[i + shift], &d)?;
        }
        r[dr as usize] = k_zero(pv);
        r = k_trim(r);
    }
    Ok((k_trim(q), r))
}

fn kp_div_exact(a: &KPoly, b: &KPoly, pv: &[ExprId]) -> Result<KPoly, ApartError> {
    let (q, r) = kp_divrem(a, b, pv)?;
    if !k_trim(r).is_empty() {
        // The callers only ever ask for divisions the theory guarantees; a
        // remainder here means the ℚ(params) bookkeeping is inconsistent, and
        // fabricating a quotient would put a wrong decomposition on the wire.
        return Err(ApartError::FactorizationFailed);
    }
    Ok(q)
}

fn kp_monic(p: &KPoly, pv: &[ExprId]) -> Result<KPoly, ApartError> {
    let d = k_degree(p);
    if d < 0 {
        return Ok(kp_zero());
    }
    let inv = k_div(&k_one(pv), &p[d as usize])?;
    kp_scale(p, &inv)
}

/// Extended Euclid over `K[var]`: `(g, s, t)` with `s·a + t·b = g`, `g` monic.
fn kp_ext_gcd(a: &KPoly, b: &KPoly, pv: &[ExprId]) -> Result<(KPoly, KPoly, KPoly), ApartError> {
    let mut r0 = k_trim(a.clone());
    let mut r1 = k_trim(b.clone());
    let mut s0 = kp_one(pv);
    let mut s1 = kp_zero();
    let mut t0 = kp_zero();
    let mut t1 = kp_one(pv);
    while !r1.is_empty() {
        let (q, r) = kp_divrem(&r0, &r1, pv)?;
        let new_s = kp_sub(&s0, &kp_mul(&q, &s1, pv)?, pv)?;
        let new_t = kp_sub(&t0, &kp_mul(&q, &t1, pv)?, pv)?;
        r0 = r1;
        r1 = r;
        s0 = s1;
        s1 = new_s;
        t0 = t1;
        t1 = new_t;
    }
    let d = k_degree(&r0);
    if d >= 0 {
        let inv = k_div(&k_one(pv), &r0[d as usize])?;
        r0 = kp_scale(&r0, &inv)?;
        s0 = kp_scale(&s0, &inv)?;
        t0 = kp_scale(&t0, &inv)?;
    }
    Ok((r0, s0, t0))
}

fn kp_mod(a: &KPoly, m: &KPoly, pv: &[ExprId]) -> Result<KPoly, ApartError> {
    let (_, r) = kp_divrem(a, m, pv)?;
    Ok(k_trim(r))
}

/// Split `num / ∏ m_i` into `Σ A_i / m_i` over pairwise-coprime moduli.
fn kp_partial_fractions(
    num: &KPoly,
    moduli: &[KPoly],
    pv: &[ExprId],
) -> Result<Vec<KPoly>, ApartError> {
    let n = moduli.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![kp_mod(num, &moduli[0], pv)?]);
    }
    let mut result = Vec::with_capacity(n);
    let mut cur = k_trim(num.clone());
    for i in 0..n - 1 {
        let mi = &moduli[i];
        let mut rest = kp_one(pv);
        for m in &moduli[i + 1..] {
            rest = kp_mul(&rest, m, pv)?;
        }
        let (_g, _s, t) = kp_ext_gcd(mi, &rest, pv)?;
        let ai = kp_mod(&kp_mul(&cur, &t, pv)?, mi, pv)?;
        let next = kp_div_exact(&kp_sub(&cur, &kp_mul(&ai, &rest, pv)?, pv)?, mi, pv)?;
        result.push(ai);
        cur = next;
    }
    result.push(cur);
    Ok(result)
}

/// `f`-adic expansion of `a/f^e` into `[A_1, …, A_e]` with `a/f^e = Σ A_j/f^j`.
fn kp_adic_expansion(
    a: &KPoly,
    f: &KPoly,
    e: u32,
    pv: &[ExprId],
) -> Result<Vec<KPoly>, ApartError> {
    let mut digits: Vec<KPoly> = Vec::with_capacity(e as usize);
    let mut cur = k_trim(a.clone());
    for _ in 0..e {
        let (q, r) = kp_divrem(&cur, f, pv)?;
        digits.push(k_trim(r));
        cur = k_trim(q);
    }
    digits.reverse();
    Ok(digits)
}

// ===========================================================================
// Conversion
// ===========================================================================

/// Read a multivariate polynomial over `gens` (with `gens[0]` the main
/// variable) as a polynomial in `gens[0]` with `ℚ(gens[1..])` coefficients.
fn mpoly_to_kpoly(p: &MultiPoly, pv: &[ExprId]) -> Result<KPoly, ApartError> {
    let mut buckets: BTreeMap<u32, TermMap> = BTreeMap::new();
    for (exp, coeff) in &p.terms {
        let d = exp.first().copied().unwrap_or(0);
        let mut rest: Vec<u32> = exp.iter().skip(1).copied().collect();
        while rest.last() == Some(&0) {
            rest.pop();
        }
        buckets.entry(d).or_default().insert(rest, coeff.clone());
    }
    let top = buckets.keys().next_back().copied().unwrap_or(0);
    let mut out: KPoly = Vec::with_capacity(top as usize + 1);
    for d in 0..=top {
        let terms = buckets.remove(&d).unwrap_or_default();
        out.push(k_from_mpoly(MultiPoly {
            vars: pv.to_vec(),
            terms,
        })?);
    }
    Ok(k_trim(out))
}

/// `K` element → expression.
fn k_to_expr(c: &K, pool: &ExprPool) -> ExprId {
    let n = c.numer.to_expr(pool);
    if is_one_mpoly(&c.denom) {
        return n;
    }
    let d = pool.pow(c.denom.to_expr(pool), pool.integer(-1_i32));
    // `1 * d^{-1}` and `d^{-1}` are the same expression to the simplifier but
    // not to the inverse-transform tables, which match on term shape.
    if n == pool.integer(1_i32) {
        return d;
    }
    pool.mul(vec![n, d])
}

fn is_one_mpoly(p: &MultiPoly) -> bool {
    p.terms.len() == 1 && p.terms.get(&Vec::new()).is_some_and(|c| *c == 1)
}

/// `K[var]` element → expression.
fn kpoly_to_expr(p: &KPoly, var: ExprId, pool: &ExprPool) -> ExprId {
    let p = k_trim(p.clone());
    if p.is_empty() {
        return pool.integer(0_i32);
    }
    let one = pool.integer(1_i32);
    let mut summands = Vec::new();
    for (d, c) in p.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let ce = k_to_expr(c, pool);
        // Omit a unit coefficient: the inverse-transform tables pattern-match
        // on the shape of these terms, and a stray `1 *` factor is a different
        // shape.
        let power = match d {
            0 => one,
            1 => var,
            _ => pool.pow(var, pool.integer(d as i32)),
        };
        let term = if ce == one {
            power
        } else if d == 0 {
            ce
        } else {
            pool.mul(vec![ce, power])
        };
        summands.push(term);
    }
    match summands.len() {
        0 => pool.integer(0_i32),
        1 => summands[0],
        _ => pool.add(summands),
    }
}

/// Lift a `ℚ[params]` polynomial to the full generator set by prepending a
/// zero exponent for the main variable.
fn lift_to_gens(p: &MultiPoly, gens: &[ExprId]) -> MultiPoly {
    let mut terms = TermMap::new();
    for (exp, coeff) in &p.terms {
        let mut key = Vec::with_capacity(exp.len() + 1);
        key.push(0u32);
        key.extend_from_slice(exp);
        while key.last() == Some(&0) {
            key.pop();
        }
        terms.insert(key, coeff.clone());
    }
    MultiPoly {
        vars: gens.to_vec(),
        terms,
    }
}

// ===========================================================================
// Side conditions
// ===========================================================================

/// Normalise a parameter polynomial to a canonical sign so `ke − ka` and
/// `ka − ke` yield one condition rather than two.
fn sign_normalize(p: MultiPoly) -> MultiPoly {
    match p.terms.iter().next_back() {
        Some((_, lc)) if *lc < 0 => -p,
        _ => p,
    }
}

/// Record `q ≠ 0` for every irreducible factor of the denominator `d` that the
/// decomposition actually divided by.
///
/// Factors that already divide the *input* denominator are skipped: the input
/// itself has a pole there (numerator and denominator are coprime by
/// construction), so the decomposition is not making a new claim about that
/// locus.
fn collect_denominator_conditions(
    d: &MultiPoly,
    gens: &[ExprId],
    input_den: &MultiPoly,
    out: &mut Vec<MultiPoly>,
) {
    if is_one_mpoly(d) || d.is_zero() {
        return;
    }
    let lifted = lift_to_gens(d, gens);
    let irreducibles = match lifted.factor_irreducible() {
        Some((_unit, fs)) => fs.into_iter().map(|(f, _e)| f).collect::<Vec<_>>(),
        // No factorization: report the whole denominator rather than nothing.
        None => vec![lifted.clone()],
    };
    for f in irreducibles {
        if f.total_degree() == 0 {
            continue; // a non-zero integer is never a hypothesis
        }
        if mpoly_exact_div(input_den, &f).is_some() {
            continue; // the input already has a pole on {f = 0}
        }
        let f = sign_normalize(f);
        if !out.contains(&f) {
            out.push(f);
        }
    }
}

fn scan_kpoly_denominators(
    p: &KPoly,
    gens: &[ExprId],
    input_den: &MultiPoly,
    out: &mut Vec<MultiPoly>,
) {
    for c in p {
        collect_denominator_conditions(&c.denom, gens, input_den, out);
    }
}

// ===========================================================================
// Entry point
// ===========================================================================

/// Partial-fraction `expr` in `var` over `ℚ(params)`.
///
/// See the [module docs](self) for the algorithm and for what the returned
/// conditions mean.  Returns [`ApartError::NotRational`] when `expr` is not a
/// rational function of `var` even over that larger field.
pub(crate) fn apart_param(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<ParamApart, ApartError> {
    let (rf, gens) =
        rational_over_generators(expr, vec![var], pool).map_err(|_| ApartError::NotRational)?;

    // Every non-`var` generator must be a genuine *parameter* — free of `var`.
    //
    // The generator machinery is happy to abstract `exp(x)` into a polynomial
    // variable, and the resulting decomposition would even be a true identity.
    // But its hypotheses would not be about parameters: the coefficient
    // denominator `exp(x) − 1` of `apart(1/((x − exp(x))(x − 1)))` vanishes at
    // `x = 0`, a puncture in the *variable*, reported in the shape of a fact
    // about a constant.  Refusing keeps `ApartError::NotRational` meaning what
    // it meant — and keeps this path to the case it was built for.
    if gens[1..]
        .iter()
        .any(|&g| !crate::integrate::risch::poly_rde::is_free_of_var(g, var, pool))
    {
        return Err(ApartError::NotRational);
    }

    let pv: Vec<ExprId> = gens[1..].to_vec();

    if rf.denom.is_zero() {
        return Err(ApartError::ZeroDenominator);
    }
    if rf.denom.degree_in(0) > MAX_PARAM_DEGREE || rf.numer.degree_in(0) > MAX_PARAM_DEGREE {
        return Err(ApartError::NotRational);
    }

    let num_k = mpoly_to_kpoly(&rf.numer, &pv)?;
    let den_k = mpoly_to_kpoly(&rf.denom, &pv)?;
    let den_deg = k_degree(&den_k);
    if den_deg < 0 {
        return Err(ApartError::ZeroDenominator);
    }

    // Track every denominator the decomposition introduces.
    let mut cond_polys: Vec<MultiPoly> = Vec::new();

    // Normalise: divide through by lc(den) so `den` is monic in `var`.
    let lead = den_k[den_deg as usize].clone();
    let num_k = kp_scale(&num_k, &k_div(&k_one(&pv), &lead)?)?;
    let den_k = kp_monic(&den_k, &pv)?;
    scan_kpoly_denominators(&num_k, &gens, &rf.denom, &mut cond_polys);

    // Degree-0 denominator: nothing to decompose.
    if k_degree(&den_k) < 1 {
        let out = kpoly_to_expr(&num_k, var, pool);
        return Ok(ParamApart {
            expr: out,
            conditions: to_conditions(&cond_polys, pool),
        });
    }

    let (quo, rem) = kp_divrem(&num_k, &den_k, &pv)?;
    scan_kpoly_denominators(&quo, &gens, &rf.denom, &mut cond_polys);

    let mut terms: Vec<ExprId> = Vec::new();
    if !k_trim(quo.clone()).is_empty() {
        terms.push(kpoly_to_expr(&quo, var, pool));
    }

    // Reconstruction accumulator: everything emitted below, put back over the
    // common denominator `den_k`.  Checked against `num_k` before returning —
    // see the note at the end of this function.
    let mut recon = kp_mul(&quo, &den_k, &pv)?;

    if !k_trim(rem.clone()).is_empty() {
        // Factor the denominator as a multivariate ℤ-polynomial.  Gauss's
        // lemma: the irreducible factors that involve `var` are exactly the
        // irreducible factors over K[var]; the rest are its ℚ[params] content,
        // already absorbed by the monic normalisation above.
        let (_unit, all) = rf
            .denom
            .factor_irreducible()
            .ok_or(ApartError::FactorizationFailed)?;
        let mut factors: Vec<(KPoly, u32)> = Vec::new();
        for (f, mult) in all {
            if f.degree_in(0) == 0 {
                continue; // parameter content, not a pole in `var`
            }
            let fk = kp_monic(&mpoly_to_kpoly(&f, &pv)?, &pv)?;
            scan_kpoly_denominators(&fk, &gens, &rf.denom, &mut cond_polys);
            factors.push((fk, mult));
        }
        if factors.is_empty() {
            return Err(ApartError::FactorizationFailed);
        }

        let mut moduli: Vec<KPoly> = Vec::with_capacity(factors.len());
        for (f, e) in &factors {
            moduli.push(kp_pow(f, *e, &pv)?);
        }

        let parts = kp_partial_fractions(&rem, &moduli, &pv)?;

        for (i, ((f, e), a_i)) in factors.iter().zip(parts.iter()).enumerate() {
            // `den_k / f_i^{e_i}` — the other factors' contribution to the
            // common denominator.
            let rest = kp_div_exact(&den_k, &moduli[i], &pv)?;
            let coeffs = kp_adic_expansion(a_i, f, *e, &pv)?;
            for (j, a_ij) in coeffs.iter().enumerate() {
                let a_ij = k_trim(a_ij.clone());
                if a_ij.is_empty() {
                    continue;
                }
                scan_kpoly_denominators(&a_ij, &gens, &rf.denom, &mut cond_polys);
                let pow = j + 1;

                // A_{ij}/f^j over the common denominator is A_{ij}·rest·f^{e−j}.
                let f_pow = kp_pow(f, *e - pow as u32, &pv)?;
                let contrib = kp_mul(&kp_mul(&a_ij, &rest, &pv)?, &f_pow, &pv)?;
                recon = kp_add(&recon, &contrib, &pv)?;

                let f_expr = kpoly_to_expr(f, var, pool);
                let num_expr = kpoly_to_expr(&a_ij, var, pool);
                let den_expr = pool.pow(f_expr, pool.integer(-(pow as i32)));
                terms.push(if num_expr == pool.integer(1_i32) {
                    den_expr
                } else {
                    pool.mul(vec![num_expr, den_expr])
                });
            }
        }
    }

    // Recombine and compare, exactly, in ℚ(params)[var].
    //
    // Nothing downstream re-checks this: `inverse_laplace_transform` maps each
    // term through a table and adds the results, so a decomposition that is not
    // equal to its input produces a wrong *function*, silently.  Every step
    // above is exact field arithmetic and every division is one the theory
    // guarantees — but "the theory guarantees it" is what the CRT split assumes
    // about coprimality, and coprimality over ℚ(params) is the thing this whole
    // module is careful about.  The check is cheap at these degrees and turns
    // any bookkeeping slip into a refusal instead of a wrong answer.
    if !kp_eq(&recon, &num_k, &pv)? {
        return Err(ApartError::FactorizationFailed);
    }

    let out = match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    };
    Ok(ParamApart {
        expr: out,
        conditions: to_conditions(&cond_polys, pool),
    })
}

fn to_conditions(polys: &[MultiPoly], pool: &ExprPool) -> Vec<SideCondition> {
    polys
        .iter()
        .map(|p| SideCondition::NonZero(p.to_expr(pool)))
        .collect()
}

#[cfg(test)]
mod tests;
